
clear all;
close all;
clc;

I=im2double(imread('testimage2.png'));
x=[96 51 98 202 272 280 182];
y=[63 147 242 262 211 97 59];
P=[x(:) y(:)];
Options=struct;
Options.Verbose=true;
Options.Iterations=400;
Options.Wedge=2;
Options.Wline=0;
Options.Wterm=0;
Options.Kappa=4;
Options.Sigma1=8;
Options.Sigma2=8;
Options.Alpha=0.1;
Options.Beta=0.1;
Options.Mu=0.2;
Options.Delta=-0.1;
Options.GIterations=600;
[O,J]=Snake2D(I,P,Options);