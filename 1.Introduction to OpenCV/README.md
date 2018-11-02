# 第一章：OpenCV简介
本章节将介绍如何在不同系统平台安装配置OpenCV。
***
## OpenCV-Python教程简介
***
### OpenCV

OpenCV于1999年由Gary Bradsky在英特尔创立，第一个版本于2000年问世。随后Vadim Pisarevsky加入了Gary Bradsky，主要负责管理英特尔的俄罗斯软件OpenCV团队。2005年，OpenCV被用于Stanley车型，并赢得2005年DARPA大挑战。后来，它在Willow Garage的支持下持续并积极发展，转为由Gary Bradsky和Vadim Pisarevsky领导该项目。OpenCV现在支持与计算机视觉和机器学习相关的众多算法，并且正在日益扩展。

OpenCV支持各种编程语言，如C ++，Python，Java等，可以在不同的系统平台上使用，包括WindowsLinux，OS X，Android和iOS。基于CUDA和OpenCL的高速GPU操作接口也在积极开发中。

OpenCV-Python是OpenCV的Python API，结合了OpenCV C++ API和Python语言的最佳特性。

### OpenCV中的Python实现

OpenCV-Python是一个Python绑定库，旨在解决计算机视觉问题。

Python是一种由Guido van Rossum开发的通用编程语言，它很快就变得非常流行，主要是因为它的简单性和代码可读性。它使程序员能够用更少的代码行表达思想，而不会降低可读性。

与C / C++等语言相比，Python速度较慢。也就是说，Python可以使用C / C++轻松扩展，这使我们可以在C / C++中编写计算密集型代码，并创建可用作Python模块的Python包装器。这给我们带来了两个好处：首先，代码与原始C / C++代码一样快（因为它是在后台工作的实际C++代码），其次，在Python中编写代码比使用C / C++更容易。OpenCV-Python是原始OpenCV C++实现的Python包装器。

OpenCV-Python使用Numpy，这是一个高度优化的数据库操作库，具有MATLAB风格的语法。所有OpenCV数组结构都转换为Numpy数组。这也使得与使用Numpy的其他库（如SciPy和Matplotlib）集成更容易。

### OpenCV-Python教程

OpenCV引入了一组新的教程，它将指导你完成OpenCV-Python中提供的各种功能。本指南主要关注OpenCV 3.x版本（尽管大多数教程也适用于OpenCV 2.x）。

建议事先了解Python和Numpy，因为本指南不涉及它们。为了使用OpenCV-Python编写优化代码，必须熟练使用Numpy。

本教程最初由Abid Rahman K.在Alexander Mordvintsev的指导下作为Google Summer of Code 2013计划的一部分启动。

### OpenCV需要你!!!

由于OpenCV是一个开源计划，欢迎所有人为图书馆，文档和教程做出贡献。如果你在本教程中发现任何错误（从一个小的拼写错误到代码或概念中的一个令人震惊的错误），请随意通过在GitHub中克隆OpenCV并提交拉取请求来纠正它。 OpenCV开发人员将检查你的请求，给你重要的反馈，并且（一旦通过审核者的批准），它将合并到OpenCV中。然后你将成为一个开源贡献者:-)

随着新模块被添加到OpenCV-Python，本教程将不得不进行扩展。如果你熟悉特定算法并且可以编写包含算法基本理论和显示示例用法的代码的教程，请执行此操作。

记住，我们在一起可以使这个项目取得圆满成功！
