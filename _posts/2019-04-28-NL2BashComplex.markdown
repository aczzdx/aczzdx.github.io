---
layout: post
title: "NL2Bash-Complex: Synthesis Complex Bash Scripts from Natural Language"
authors: "Changlin Zhang, Chenguang Hong, Chiyu Song, Wei-Lun Hsiao, Xingyu Wei"
---

# Introduction

Shell scripting is an essential skill for many PC users and server maintainers. However, learning shell scripts could be painful. Although the name of the utilities is easy to understand, it is difficult to memorize all the way to use the flags and arguments for each utility. There is [a GitHub project](https://github.com/nvbn/thefuck) that helps the users to correct the previously failed command. Nonetheless, this project is rule-based, and its performance cannot be satisfied when the script is too long. Users still need a lot of time to construct correct bash commands, even if they are similar to what they typed yesterday. 

In this project, we worked on the problem converting descriptions in natural language into shell commands in bash. Our goal is to extend the state-of-the-art baseline model to output commands with complex structures and achieve better performance. We achieve it by specifying pretrained word-embedding model and adding TypeSelector structure.

# From Natural Language to Bash Scripts

The problem we are going to solve is like machine translation: given a task description $X$ as the input, we synthesis the correspondent program, $P$, in bash script that is executable and able to fulfill the task. 

<center><em>Illustration of an example in the dataset</em></center>

![Example of the problem ](/images/example.png)

Lin et al. collected an open-source dataset, [NL2Bash](https://github.com/TellinaTool/nl2bash),  with more than ten thousand manually verified (Natural language, Bash Scripts) pairs. They also provide state-of-the-art prediction model.

# Seq2Seq or Seq2Tree

<!---
Before the study in generating bash scripts, lots of researchers have been working on translating from natural language into regular expressions [2], SQL queries [3, 4], and etc. Interesting work from Ling et al [6] and Rabinovich et al [7] aims to parse the descriptions of the cards in Hearthstone into class definitions in Python. What is more, Yin and Neubig in [8] and Nye et al in [9] purpose neural architectures for general text-to-code tasks. --->

Most of the works in program synthesis use a stage-wise strategy. The main idea is to predict the abstract syntax tree with “slots” representing the name and literal first, and then fill in the slots given the information from the input description. Another strategy is to treat the programming language as a natural language and use Seq2Seq model, which is already widely used in machine translation. Lin in NL2Bash compared the stage-wise model with an extended Seq2Seq model, [CopyNet](https://arxiv.org/abs/1603.06393), and found the latter model performed better in the dataset of shell commands.

The baseline model we are comparing is the CopyNet model in NL2bash, which is using the copy mechanism in Seq2Seq model to solve Out-of-Vocabulary problem by adding predictions on copying tokens from encoder input directly into the output. This is also the state-of-the-art model for NL2Bash which has better performance over stage-wise strategies. 

<center><em>The architecture of the decoder in the baseline model</em></center>
![The architecture of the decoder in the baseline model, which are using two attentions score to get predict from the target reference set and encoder inputs.](/images/decoder.png "The architecture of the decoder in the baseline model")


However, we find that this Seq2Seq-based model may make mistakes when the task needs other structure. For example, in the testing set, there is an example whose task description is 
>  Find all .js files in the $d directory tree whose path-names do not contain whitespaces 

The ground-truth code is 
```{bash} 
find $d -name ’*.js’ | grep -v " " 
```
which is using `find` to get all js files and `grep` to filter out the answer containing whitespaces. However, the best prediction from the baseline is 
```{bash} 
find "$d" -name "*.js" !  -path "$d" -prune -or -name "*.js" 
```

which just output code can find all js files in the given directory with no filter. We find that the output code is using `-prune` flag to exclude something, but it seems that it does not match the requirement of the task description. 

In summary, it seems that the baseline model is suffering from long task description or involving complicated bash script grammar. Thus, here we define the commands calling only one utility as **simple commands** while the others are **complex commands**. 
And we are trying to improve the model to perform better, especially when the output should be a complex command.



# First Step to Improve: Add Pretrained Model
The first approach we took is to use pre-trained embedding weights for the embedding layer after the encoder input. The following figure shows the process of how to obtain the embedding dictionary. First, we use [crawler4j](https://github.com/yasserg/crawler4j) to extract 10000 webpages from Wikipedia, [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) to extract the text and [NLTK](https://www.nltk.org/) to clean the data. Then, we use [Glove](https://github.com/stanfordnlp/GloVe) to train the embedding dictionray from the input of training set and the Wikipedia pages. Finally, we use the pre-trained model to substitute the embedding process before the encoder.

<center><em>The training process of embedding dictionary</em></center>

![Training Process](/images/embedding.png)



# Another Step to Improve: TypeSelector
In the previous example illustrating drawbacks of NL2bash baseline model, we find that dealing with various kinds of commands could be an essential task for a better model to solve. In our best work, we tried to train different decoders dedicated to different scenarios, and before the encoder outputs are fed to the decoder, we use TypeSelector, a classifier implemented with the feed-forward network, to predict whether the output should be a simple command or a complex one. Then we ran the decoders at the same time and mask out the final output.

<center><em>The architecture of the model using TypeSelector</em></center>
![The architecture of the model using TypeSelector](/images/model.png)

Meanwhile, to train the model, we modify the loss function to control the prediction of the TypeSelector. The loss function for the TypeSelector is the cross-entropy between the type of the ground-truth bash code, simple or complex, and the predicted type, which is

$$ L_T(\mathrm{t}, \mathrm{q}) = - \frac{1}{N} \sum_{i=1}^{N} \left(t_i \log(q_i) + (1- t_i) \log(1 - q_i) \right) $$

where $N$ is the number of samples, $t_i$ is the ground truth label for the $i$th example and $q_i$ is the probability that the output of $i$th input is simple.
While training, we add the new loss to the original token-wise cross-entropy multiplied by $\gamma$ as a hyperparameter to weight the result.

$$ L = L_{\mathrm{orignial}} + \gamma L_T(\mathrm{t}, \mathrm{q}) $$

We also tried to modify a general text-to-code model, [TRANX](https://github.com/pcyin/tranX), to handle bash scripts. However, its current performance is much worse than our best work.

# Yet Another Step to Improve: Add Semantic Loss

The intuition to try to add semantic loss is inspired by [Variational Autoencoder(VAE)](https://arxiv.org/abs/1312.6114). VAE adds an additional latent loss to constrain the model’s latent variables’ distribution. It makes us consider adding a regularization item. [Semantic loss](http://web.cs.ucla.edu/~guyvdb/papers/XuICML18.pdf) is a good possible add-on. Its design intention is to capture logical constraints and structured predictions, like one-hot constraint when encoding in addition to standard sigmoid cross entropy. Semantic loss can be formualated as the following.

$$L^s (\alpha, \mathbf{p}) \propto - \log \sum_{x \models \alpha} \prod_{i: \mathbf{x} \models X_i} p_i \prod_{i: x \models \neg X_i} (1 - p_i)$$

where $\mathbf{p}$ is the vector of possibilities for each variable in $\mathbf{x}$, and $\alpha$ is a sentence over $\mathbf{x}$. Then, we add semantic loss to the orignial loss function and control it with hyperparameter $\omega$. 

$$ L = L_{\mathrm{orignial}} + \omega \, L^s(\alpha, \mathbf{p})$$

# Experiment and Evaluation

To experiment, we construct two tests set to evaluate the performance of the model. One of them is the original test set provided by NL2Bash, and the other one is consist of commands that are all complex. To test the performance of each approach, we keep the remaining hyperparameters in the model to be the same ones as the baseline model.

The metrics we used are Template Matching score \(TM\), which is the coverage of the overlapping tokens between the ground truth and the predictions, and BLEU score, which is widely used in machine translation model.  Also, since we use beam search to find multiple predictions, we use superscript $k$ to indicate the best score could be achieved if considering the top $k$ predictions from the models.

The following tables show the evaluation results for two values on two test sets. 



<center><em>Evaluation result on the test set provided by NL2Bash</em></center>

| Model (Approach) | TM<sup>1</sup> | BLEU<sup>1</sup> | TM<sup>3</sup> | BLEU<sup>3</sup> |
|------------------|----------------|------------------|----------------|------------------|
| TypeSelector     | **0.625**      | **0.563**        | **0.707**      | **0.638**        |
| Pre-trained      | 0.566          | 0.497            | 0.633          | 0.570            |
| Semantic-Loss    | 0.541          | 0.475            | 0.605          | 0.542            |
| Baseline         | 0.601          | 0.543            | 0.676          | 0.619            |



<center><em>Evaluation result on the test set with complex commands only</em></center>

| Model (Approach) | TM<sup>1</sup> | BLEU<sup>1</sup> | TM<sup>3</sup> | BLEU<sup>3</sup> |
|------------------|----------------|------------------|----------------|------------------|
| TypeSelector     | **0.521**      | **0.451**        | **0.634**      | **0.537**        |
| Pre-trained      | 0.519          | 0.445            | 0.616          | 0.529            |
| Semantic-Loss    | 0.444          | 0.375            | 0.525          | 0.477            |
| Baseline         | 0.505          | 0.429            | 0.596          | 0.514            |

In this result, we find that TypeSelector improves the performance of the model in both the test-set mixing simple commands and complex commands and the one with complex commands only. The model using pre-trained dictionary has better performace than the baseline model in the complex-commands test set but failed to achieve better performance in the original test set. 

Meanwhile, the experiment results also show that adding semantic loss hurts the performance a little bit. Thus, we do not combine this method to the final-stage model. The results also agree with the orignial paper that semantic loss is more suitable for the semi-supervised scene and is not good at the full-supervised scene since semantic loss combat the standard sigmoid cross entropy loss.

# Error Analysis
We also analyze the prediction from the results of our approaches to finding whether our model has better performance in generating complex commands. We selected some typical results to demonstrate the comparison between the baseline model and the output from the best model. 

Firstly, the following table shows the model predictions over the task we mentioned in the section of the baseline model. In this example, we see that all three predictions from the baseline failed to output a prediction that having the correct results. However, in the model using TypeSelector, all three predictions use pipeline and figure out to filter out the correct candidates by calling `grep`.

<table>
  <tr>
<td style="text-align: center"> <em>Directions</em> </td>
<td>Find all .js files in the $d directory tree whose pathnames do not contain whitespaces</td>
</tr>
  <tr>
    <td style="text-align: center"><em>Ground Truth</em></td>
    <td><code>find $d -name ’*.js’ | grep-v " " </code></td>
  </tr>
  <tr>
    <td style="text-align: center" rowspan="3"><em>TypeSelector Results</em></td>
    <td style="text-align: left">
        <code>find $d -name ’*.js’ | grep -v whitespaces</code> </td>
  </tr>
  <tr>
  <td>
  <code>
    find $d -name ’*.js’ | xargs -I {} grep ’whitespaces’ {}<br/>
    </code>
    </td>
  </tr>
  <tr><td>  <code>find $d -name ’*.js’ | grep -v ’whitespaces’</code> </td></tr>
  <tr>
    <td style="text-align: center" rowspan="3"><em>Baseline Results</em></td>
    <td style="text-align: left"><code class="language-{bash}" style="text-align: left">
    find "$d" -name "*.js" !  -path "$d" -prune -or -name "*.js"
    </code>
    </td></tr>
  <tr><td><code>
    find "$d" -name "*.js" !  -path "$d" -prune -or -name "*.js"
  </code></td></tr>
    <tr><td><code>
    find "$d" -name "*.js" !  -path "$d" -prune -or -name ’*.js<br/>
    </code></td></tr>
  
</table>

The following table is another example of generating complex codes. In this example, the model using TypeSelector have all its top-3 predictions using pipeline and have the same correct output. On the other hand, although the baseline model is able to predict the correct code, the model does not prioritize the correct result as the top-1 prediction, and the other two predictions in top-3 predictions is not executable. 

<table>
  <tr>
<td style="text-align: center"> <em>Directions</em> </td>
<td> Print 2 lines of "123456789".</td>
</tr>
  <tr>
    <td style="text-align: center"><em>Ground Truth</em></td>
    <td><code>
        yes 123456789 | head -2
    </code></td>
  </tr>
  <tr>
    <td style="text-align: center" rowspan="3"><em>TypeSelector Results</em></td>
    <td style="text-align: left">
        <code>
        yes 123456789 | head -n 2
        </code> </td>
  </tr>
  <tr>
  <td>
  <code>
    yes "123456789" | head -n 2
    </code>
    </td>
  </tr>
  <tr><td>  <code>
    yes ’123456789’ | head -n 2
  </code> </td></tr>
  <tr>
    <td style="text-align: center" rowspan="3"><em>Baseline Results</em></td>
    <td style="text-align: left"><code class="language-{bash}" style="text-align: left">
tail -n +2 123456789 | head -n 2 
    </code>
    </td></tr>
  <tr><td><code>
yes 123456789 | head -n 2 
  </code></td></tr>
    <tr><td><code>
tail -n +2 123456789 | head - 2 
    </code></td></tr>
  
</table>


# Next step to take

In conclusion, our contributions are: 
1. collecting corpus to customize pre-trained embedding model,
2. adding typer-selector to enable model to predict complex structure.

We find that using a simple classifier to deal with simple commands and complex commands could help improve the result of the prediction. Also, the improvement can be observed in both the overall test set and the complex command test sets. We are planning to find whether we could introduce more complicated model so that the predictor can support more kind of structures that involving control flows, like loop or conditions.

However, we still find that it is still hard to find a way to measure the accuracy of the output code. Finding a more efficient but sound test way is still a challenge in this area. 

[^zen]: [http://www.catb.org/~esr/writings/unix-koans/ten-thousand.html](http://www.catb.org/~esr/writings/unix-koans/ten-thousand.html)

[^util]: [(https://github.com/nvbn/thefuck)](https://github.com/nvbn/thefuck)

[^NL2Bash]: Xi Victoria Lin, Chenglong Wang, Luke S. Zettlemoyer, and Michael D. Ernst. Nl2bash: A corpus and semantic parser for natural language interface to the linux operating system. CoRR, abs/1802.08979, 2018. See also [https://github.com/TellinaTool/nl2bash](https://github.com/TellinaTool/nl2bash)

[^copynet]: Jiatao Gu, Zhengdong Lu, Hang Li, and Victor O. K. Li. Incorporating copying mechanism in sequence-to-sequence learning. CoRR, abs/1603.06393, 2016.

[^Bs]: [https://www.crummy.com/software/BeautifulSoup/](https://www.crummy.com/software/BeautifulSoup/)

[^crawler4j]: [https://github.com/yasserg/crawler4j](https://github.com/yasserg/crawler4j)

