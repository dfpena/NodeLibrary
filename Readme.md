![STREMECOVER](http://go.pluricorp.com/websitemedia/gitlab/templatetop.svg)

# StremeCoder BASICS

## Install StremeCoder

StremeCoder is a node Editing IDE for python and provided by Pluri Inc. It dramatically increases your code maintainability and documentability. 

You can get it [Get It Here](http://do.pluricorp.com/page/stremecoder) 

## Install Python 3

The StremeCoder does not require the presence of python to compile node graphs to python, but you will need python 3 to run any compiled file.

There are many different package managers for python, but we prefer anaconda as its packages are well mantained and offers many learning tools to make python management easy.

[Anaconda](https://www.anaconda.com/distribution/)

Make Sure to install Python 3+ as Python 2.X is no longer maintained

## Learning Python

There are many resources to start learning Python on the internet in particular for  Data Science. We suggest you begin by drawing out a simple algorithm that loads some Data, does some manipulation then saves that data. We provide these nodes and you can look through how this is accomplished in code. As our Library of nodes grows more examples and possible algorithms will be available to you. You will become comfortable very quickly with designing koi at which point you can contact Pluri about implementing that code or a designer you know. Within

## StremeCoder caveats

The StremeCoder uses special characters to insure that you can have duplicate functions that share the same python function name, but may have slight differences in code.

```
functionName|||
```

Where ||| is an enumerator that assigns a distinct number to every node that uses ||| such that a node that saves to excel with a user defined location can do it again at some later point in the Koi to different location.

In nodes with user inputs and extra pins, defining a dictionary which is normally ``` newDictionary = {"key":value} ``` will require an extra {} enclosure ``` newDictionary = {{"key":value}} ``` as extra pins use {pinName} to place node functions and {userInputKey} to place user dialog data. 

## The Flow of Streme

The StremeCoder is meant to make your algorithms readable to anyone. You should keep in mind these axioms when you are designing your algorithms.

- Generate Descriptive Node Names
- A Node that can be used by ANYONE is significantly more valuable than a Node that can only be used in your algorithm
- A Beautiful Koi(Node Graph) is elegant and can be read like a story of reasonable functions  
- Maintain, when possible, the functional flow of Streme. That is avoid placing functions and classes outside of your node functions unless no other solution is possible.
- When adding to ``` kwargs[Settings] ``` enclosing your dictionary items inside a very specific subname will avoid collisions with other nodes you may download on the internet or if you choose to make your node available to others ``` kwargs[Settings][LabNameNumber][SettingVariable] ```

## The Canvas

We intend for you to use the Canvas as real workspace. You can zoom out of the canvas (CTRL +/- and CTRL Mouse wheel) and hide nodes you want to use later. Don't worry about a messy canvas when you are working, having extra parts is useful when you are editing and duplicating nodes. 
This has been our general workflow for generating algorithms:

1. Generate a Sketch of different nodes that take you from some input through some analysis to some output.
2. Export that sketch to your client as either a Koi or as an image in an email.
3. Once everyone is in agreement with the flow of the algorithm, code the algorithm.
4. Update the descriptions by filling out the template. 
5. Potentially uploading your nodes to github, sending your koi or python files to your client

### You don't need to be coding python to use the canvas! You can export your KOI as an editable SVG. You can draw experimental designs, processes, algorithms etc.. to use in presentations.

## Making Koi

A node graph created by stremeCoder is called a Koi. A Koi is a readable text json formatted to contain all of contents of your designed node graph. Koi can be created by saving your node graphs, a buffered version of your node graph is automatically saved when changes are made to the canvas. This means when you have a single StremeCoder window open your progress will not be lost after closing. 

## Node Library

Over the course of designing a koi you will develop your own Nodes that you will want to start tracking on thier own. To do this we suggest you clear out all other nodes in your koi and rename your Koi to reflect the name of the node you want to save and save the node to a node library and upload your node to github. You should keep track of versioning of this node and import this prestine node into new projects, rather than importing projects with a node you want
