---
layout: post
title: "From Trivum to Triva"
authors: "X. Wei"
---

# TL; DR

Just some scribble notes from lots of resources, e.g.
- Hacker News
- Lectures
- Articles
- etc.

# September 2019

## The Line Breaker of HTTP Message Header Is CRLF in RFC2616

In [RFC2616](https://www.w3.org/Protocols/rfc2616/rfc2616-sec4.html) 
we may find this interesting definition of HTTP message:

        generic-message = start-line
                          *(message-header CRLF)
                          CRLF
                          [ message-body ]
        start-line      = Request-Line | Status-Line

It could be a little bit confusing if someone wants to write a HTTP
message parser, or compose a HTTP message manually in Unix-like OS
since the deafult line breaker is LF(`\n`), which stands for **L**ine **F**eed,
that without a preceeding CR(`\r`), which stands for **C**arriage **R**eturn.

However, from 
[this Quora question](https://www.quora.com/Why-does-the-HTTP-spec-use-r-n-CR-LF-to-separate-headers-instead-of-n-LF)
we find that RFC2616 recommends applications recongnize a single LF and ignore the preceeding CR.

That's good news. :joy:
