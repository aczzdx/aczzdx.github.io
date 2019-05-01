---
layout: post
title: "NL2Bash++: Synthesis Complex Bash Scripts from Natural Language"
authors: "Changlin Zhang, Chenguang Hong, Chiyu Song, Wei-Lun Hsiao, Xingyu Wei"
---

# Introduction

Shell scripting is an essential skill for many PC users and server maintainers. However, learning shell scripts could be painful. Although the name of the utilities is easy to understand, it is difficult to memorize all the way to use the flags and arguments for each utility. There is [a GitHub project](https://github.com/nvbn/thefuck) that helps the users to correct the previously failed command. Nonetheless, this project is rule-based, which need efforts to collect rules manually, and its performance cannot be satisfied when the script is too long. Users still need a lot of time to construct correct bash scripts, even if they are similar to what they typed yesterday.


Apart from a rule-based solution, researchers in program synthesis are attempting to use deep learning to generate code from descriptions in natural language. They are capable of using deep learning to translate task instructions into [regular expressions](https://www.aclweb.org/anthology/N13-1103),  [SQL queries](https://arxiv.org/abs/1804.08338), etc. Also, [Rabinovich et al](https://arxiv.org/abs/1704.07535) are capable of parsing the descriptions of the cards in Hearthstone into class implementation in Python. Compared to the traditional rule-based solution, machine learning can save time for experts to compose rules to generate codes. Applying machine learning in forming bash scripts could be promising to improve the status quo.

In this project, we worked on the problem converting descriptions in natural language into shell scripts in bash. Our goal is to extend the state-of-the-art baseline model, [NL2Bash](https://github.com/TellinaTool/nl2bash), to output scripts with complex structures and achieve better performance. After analysis the result from the baseline model, we find that although the baseline model performs well for lots of cases, it still suffers from some problems that we attempt to address. 

- When the input involving name entities, e.g. numbers, directory names, etc., the baseline model will copy them directly into the output rather than translate them into the correct form. For example, "root directory" should be translated into `/`. 
- When the code needs to use grammar structures, e.g., pipeline or subshell, and multiple bash utilities,  the baseline model tends to output code that using one utility.
- The output of the baseline model is vectors of probability over the words in the given vocabulary and it may not follow some logical constraints, e.g., only one of them should be near 1 and the others should be near 0.

We attempt to address them by specifying pretrained word-embedding model, adding TypeSelector structure and adding semantic loss, respectively. 

# From Natural Language to Bash Scripts

The problem we are going to solve is like machine translation: given a task description $X$ as the input, we synthesis the correspondent one-line program, $P$, in bash script that is executable and able to fulfill the task. 

<center><em>Illustration of an example in the dataset</em></center>

![Example of the problem in NL2Bash](/images/example.png)

Lin et al. collected an open-source dataset, [NL2Bash](https://github.com/TellinaTool/nl2bash), with more than ten thousand manually verified (Natural language, Bash Scripts) pairs. The dataset exclude the statements calling non-bash program (`awk`, `sed`, `python`, `eamcs`, `vi`, and etc.), all compound statements with control-flow keywords (`if`, `for`, `while`, `until`), functions, and redirections.

They also provide state-of-the-art prediction model, which can output one-line code from the description in natural language and we will discuss in the next section.

# Baseline model and its drawback
<!---
Before the study in generating bash scripts, lots of researchers have been working on translating from natural language into regular expressions [2], SQL queries [3, 4], and etc. Interesting work from Ling et al [6] and Rabinovich et al [7] aims to parse the descriptions of the cards in Hearthstone into class definitions in Python. What is more, Yin and Neubig in [8] and Nye et al in [9] purpose neural architectures for general text-to-code tasks. --->

Most of the works in program synthesis use a stage-wise strategy. The main idea is to predict the abstract syntax tree with “slots” representing the name and literal first, and then fill in the slots given the information from the input description. Another strategy is to treat the programming language as a natural language and use Seq2Seq model, which is already widely used in machine translation. Lin in NL2Bash compared the stage-wise model with an extended Seq2Seq model, [CopyNet](https://arxiv.org/abs/1603.06393), and found the latter model performed better in the dataset of shell commands.

The baseline model we are comparing is the CopyNet model in NL2bash, which is using the copy mechanism in Seq2Seq model to solve Out-of-Vocabulary problem by adding predictions on copying tokens from encoder input directly into the output. This is also the state-of-the-art model for NL2Bash which has better performance over stage-wise strategies. 

<center><em>The architecture of the decoder in the baseline model</em></center>
![The architecture of the decoder in the baseline model, which are using two attentions score to get predict from the target reference set and encoder inputs.](/images/decoder.png "The architecture of the decoder in the baseline model")

However, we find that the baseline model still suffers from some problems in the introduction.
For example, in the testing set, there is an example whose task description is 

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

Meanwhile, if we change "directory" into "folder" or other synonyms, the output should be the same. However,
since the training set is small, when a word not in the training set shows up, the baseline model will treat
it as an unknown token and copy it to the output directly rather than translate it into correct argument or flags.

What is more, the output of the model is a vector of probability which token in the decoder reference set and the input should be used.
Sometimes there could be ties and hurt the output correctness.

Thus, we use three approaches to solve these problems. We will elaborate on the problem analysis and the approach we have taken in the following sections.
    
# Understand NL better: Pre-trained Word-embedding Model

The reason why we add pre-trained embedding model is to help the model to understand other words that are not shown in the training set. Compared to other works in natural language processing, the size of the training set in NL2Bash is much smaller. It is unlikely that the word-embedding layer in the original model could learn the natural language properly. Using a pre-trained word-embedding model may mitigate this problem.

In this work, we use pre-trained embedding weights for the embedding layer after the encoder input. The following figure shows the process of how to obtain the embedding dictionary. First, we use [crawler4j](https://github.com/yasserg/crawler4j) to extract 10000 webpages from Wikipedia, [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) to extract the text and [NLTK](https://www.nltk.org/) to clean the data. Then, we use [Glove](https://github.com/stanfordnlp/GloVe) to train the embedding dictionary from the input of training set and the Wikipedia pages. Finally, we use the pre-trained model to substitute the embedding process before the encoder.

<center><em>The training process of embedding dictionary</em></center>

![Training Process](/images/embedding.png)


# Predict Complex Scripts Separately: TypeSelector
The intuition we add TypeSelector is inspired when we find that the baseline model may output the script use one utility only when the task needs calling multiple Bash utilities and using bash grammar like pipelines or subshells. To demonstrate the problem clearly, we define two types of Bash script according to the [Bash manual](https://linux.die.net/man/1/bash).

- **Simple scripts**: A sequence of blank-separated words, a sequence of characters not containing any character used for control flow. It is similar to the definition of simple commands in the Bash manual, but we restrict the use of redirections and control character at the end of the command.
It is similar to the definition of simple commands in the Bash manual, but we restrict the use of redirections and control character at the end of the command.

- **Complex scripts**: Codes containing of multiple simple scripts, which are combined by control operators, e.g. pipe (`|`), logical operators (`&&`, `||`), etc.

The following table shows examples of simple commands and complex commands. 

<center><em> Examples of simple scripts and complex scripts </em></center>

<table>
  <tr>
    <th style="text-align center">Simple</th>
    <td style="text-align center"><code>rm -f foo.txt</code></td>
  </tr>
  <tr>
    <th style="text-align center">Complex</th>
    <td style="text-align center">
    <code>yes x | head -n 10</code></td>
  </tr>
</table>

From the example analyzing the baseline model at the beginning of this section, we find that the baseline model does not perform well when predicting the complex scripts. Dealing with different types of scripts could be an essential task a better model should solve. In this work we propose TypeSelector, a classifier implemented with the feed-forward network, to predict whether the output should be a simple script or a complex one. Then we ran the decoders at the same time and mask out the final output.

<center><em>The architecture of the model using TypeSelector</em></center>
![The architecture of the model using TypeSelector](/images/model.png)

Meanwhile, to train the model properly, we modify the loss function to control the prediction of the TypeSelector. The loss function for the TypeSelector is the cross-entropy between the type of the ground-truth bash code, simple or complex, and the predicted type, which is

$$ L_T(\mathrm{t}, \mathrm{q}) = - \frac{1}{N} \sum_{i=1}^{N} \left(t_i \log(q_i) + (1- t_i) \log(1 - q_i) \right) $$

where $N$ is the number of samples, $t_i$ is the ground truth label for the $i$th example and $q_i$ is the probability that the output of $i$th input is simple.
While training, we add the new loss to the original token-wise cross-entropy multiplied by $\gamma$ as a hyperparameter to weight the result.

$$ L = L_{\mathrm{original}} + \gamma L_T(\mathrm{t}, \mathrm{q}) $$

Also, we attempted to modify a general text-to-code model, [TRANX](https://github.com/pcyin/tranX), to handle bash scripts. However, its current performance is much worse than our best work. In addition, we tried to train TypeSelector and Encoder-Decoder separately and combined them together afterwards. However, the result is worse because we cannot promise the prediction of TypeSelector is always correct and the misclassification will harm the output significantly.

# Logical Restrictions on Output: Semantic Loss

Adding semantic loss is inspired by [Variational Autoencoder(VAE)](https://arxiv.org/abs/1312.6114). VAE adds an additional latent loss to constrain the model’s latent variables’ distribution. It makes us consider adding a regularization item. [Semantic loss](http://web.cs.ucla.edu/~guyvdb/papers/XuICML18.pdf) is a good possible add-on. Its design intention is to capture logical constraints and structured predictions, like one-hot constraint when encoding in addition to standard sigmoid cross entropy. Semantic loss can be formulated as the following.

$$L^s (\alpha, \mathbf{p}) \propto - \log \sum_{x \models \alpha} \prod_{i: \mathbf{x} \models X_i} p_i \prod_{i: x \models \neg X_i} (1 - p_i)$$

where $\mathbf{p}$ is the vector of possibilities for each variable in $\mathbf{x}$, and $\alpha$ is a sentence over $\mathbf{x}$. Then, we add semantic loss to the original loss function and control it with hyperparameter $\omega$. 

$$ L = L_{\mathrm{original}} + \omega \, L^s(\alpha, \mathbf{p})$$

# Experiment and Evaluation

To experiment, we construct two tests set to evaluate the performance of the model. One of them is the original test set provided by NL2Bash, and the other one is consist of complex scripts only. To test the performance of each approach, we keep the remaining hyperparameters in the model to be the same ones as the baseline model.

The metrics we used are Template Matching score \(TM\), which is the coverage of the overlapping tokens between the ground truth and the predictions, and [BLEU score](https://dl.acm.org/citation.cfm?id=1073135) , which is widely used in machine translation model.  Also, since we use beam search to find multiple predictions, we use superscript $k$ to indicate the best score could be achieved if considering the top $k$ predictions from the models.

The following tables show the evaluation results for two values on two test sets. 

<center><em>Evaluation result on the test set provided by NL2Bash</em></center>

| Model (Approach)           | TM<sup>1</sup> | BLEU<sup>1</sup> | TM<sup>3</sup> | BLEU<sup>3</sup> |
|:--------------------------:|:--------------:|:----------------:|:--------------:|:----------------:|
| TypeSelector + Pre-trained | 0.605          | 0.543            | 0.682          | 0.625            |
| TypeSelector               | **0.621**      | **0.560**        | **0.691**      | **0.626**        |
| Pre-trained                | 0.566          | 0.497            | 0.633          | 0.570            |
| Semantic-Loss              | 0.541          | 0.475            | 0.605          | 0.542            |
| Baseline                   | 0.601          | 0.543            | 0.676          | 0.619            |



<center><em>Evaluation result on the test set with complex scripts only</em></center>

| Model (Approach)           | TM<sup>1</sup> | BLEU<sup>1</sup> | TM<sup>3</sup> | BLEU<sup>3</sup> |
|:--------------------------:|:--------------:|:----------------:|:--------------:|:----------------:|
| TypeSelector + Pre-trained | **0.539**      | 0.431            | **0.620**      | 0.526            |
| TypeSelector               | 0.530          | **0.453**        | 0.593          | 0.516            |
| Pre-trained                | 0.519          | 0.445            | 0.616          | **0.529**        |
| Semantic-Loss              | 0.444          | 0.375            | 0.525          | 0.477            |
| Baseline                   | 0.505          | 0.429            | 0.596          | 0.514            |

In this result, we find that TypeSelector improves the performance of the model in the test-set mixing simple scripts and complex scripts and the one with complex scripts if we consider the top-1 output only. The model using TypeSelector has the best performance among other models on the orignal testing set.

The model using pre-trained word-embedding model has better performance than the baseline model in the complex-scripts testing set and perform better then TypeSelector when we consider top-3 output on the complex-script-only test set. The combined model using both TypeSelector and pre-trained word-embedding method achieve higher Template Mathcing score in the complex-scripts testing set. However, both models cannot achieve better performance than the model using Typeselector only in the original test set. 

Meanwhile, the experiment results also show that adding semantic loss hurts the performance. Thus, we do not combine this method to the final-stage model. The results also agree with the original paper that semantic loss is more suitable for the semi-supervised scene and is not good at the full-supervised scene since semantic loss combat the standard sigmoid cross entropy loss.

# Error Analysis
We also analyze the prediction from the results of our approaches to finding whether our model has better performance in generating complex scripts. We selected some typical results to demonstrate the comparison between the baseline model and the output from the best model in the testing set, the model using TypeSelector only.
 
Firstly, the following table is an example of generating complex codes. In this example, the model using TypeSelector have predicted three executable codes and the first prediction is very similar to the ground truth. Although it will to print one line with 10 `x`, the code can be easily fixed by removing `tr -d '\n'`. On the other hand, the baseline model failed to output scripts that can print something. The results of the baseline model involve the action to print 10 lines, but fail to generate code to print `x`.

<table>
  <tr>
<td style="text-align: center"> <em>Directions</em> </td>
<td> Print 10 lines of "x".</td>
</tr>
  <tr>
    <td style="text-align: center"><em>Ground Truth</em></td>
    <td><code>
        yes x | head -n 10
    </code></td>
  </tr>
  <tr>
    <td style="text-align: center" rowspan="3"><em>TypeSelector Results</em></td>
    <td style="text-align: left">
        <code>
        yes 'x' | head -n 10 | tr -d '\n'
        </code> </td>
  </tr>
  <tr>
  <td>
  <code>
  x | head -n 10
    </code>
    </td>
  </tr>
  <tr><td>  <code>
  echo 'x' | head -n 10 | tr -d '\n'
  </code> </td></tr>
  <tr>
    <td style="text-align: center" rowspan="3"><em>Baseline Results</em></td>
    <td style="text-align: left"><code class="language-{bash}" style="text-align: left">
seq 10 | paste -s -d ' '
    </code>
    </td></tr>
  <tr><td><code>
head -n 10 | tr '\n' '\n'
  </code></td></tr>
    <tr><td><code>
seq 10 | xargs -I {} head -n 10 {} 
    </code></td></tr>
  
</table>

The following table is another result comparing predictions between baseline model and the model using TypeSelector. In this example, we find that the all three predictions from baseline model are scripts use `find` only. Meanwhile, all the predictions from the model using TypeSelector contain pipeline, which is near the ground-truth code. However, this example shows that both the baseline model and the one using TypeSelector cannot always capture all infromation at the same time. All predictions from both models do not contain inforamation about 50KB and searching in the sub-directories at the same time. Notice that the first prediction from the model using TypeSelector capture "50" from the description but misinterpret it as less than 50 mininutes. 

<table>
  <tr>
<td style="tet-align: center"> <em>Directions</em> </td>
<td>
 Force delete all jpg files in current directory which are less than 50KB and do not search in the sub directories
</td>
</tr>
  <tr>
    <td style="text-align: center"><em>Ground Truth</em></td>
    <td><code>
    find . -maxdepth 1 -name "*.jpg" -size -50k | xargs rm -f
    </code></td>
  </tr>
  <tr>
    <td style="text-align: center" rowspan="3"><em>TypeSelector Results</em></td>
    <td style="text-align: left">
        <code>
find . -mindepth 1 -mmin -50 -name "*.jpg" | xargs -I {} rm {} 
        </code> </td>
  </tr>
  <tr>
  <td>
  <code>
find . -mindepth 1 -maxdepth 1 -type f | xargs -I {} rm {} 
    </code>
    </td>
  </tr>
  <tr><td>  <code>
 find . -mindepth 1 -maxdepth 1 -type f | xargs -I {} rm -f {}
  </code> </td></tr>
  <tr>
    <td style="text-align: center" rowspan="3"><em>Baseline Results</em></td>
    <td style="text-align: left"><code class="language-{bash}" style="text-align: left">
find . -name "*.jpg" -size -50k -exec rm {} \; 
    </code>
    </td></tr>
  <tr><td><code>
find . -name '*.jpg' -size -50k -exec rm {} \; 
  </code></td></tr>
    <tr><td><code>
find . -name "*.jpg" -maxdepth 1 -exec rm {} \; 
    </code></td></tr>
</table>


# Next step to take

In conclusion, our contributions are: 
1. collecting corpus to customize pre-trained embedding model,
2. adding typer-selector to enable model to predict complex structure.

We find that using a simple classifier to deal with simple scripts and complex scripts could help improve the result of the prediction. Also, the improvement can be observed in both the overall test set and the complex script test sets. However, we still find the model using Seq2Seq are not able to extract all information when the description is long. We will try to solve this problem by adapting Seq2Tree models to use Seq2Seq model for predicting fragments of codes.

Also, we are planning to find whether we could introduce more complicated model so that the predictor can support more kind of structures that involving control flows, like loop or conditions.

What is more, we still find that it is still hard to find a way to measure the accuracy of the output code. Finding a more efficient but sound test way is still a challenge in this area.

