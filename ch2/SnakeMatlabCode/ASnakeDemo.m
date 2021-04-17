
%%%%    ����Snakeģ�͵Ĵ���

clear all;
close all;
clc;

%%%%    ��ʼ��ͼ���Լ���
I = imread('testimage.png');
I = im2double(I);
y = [182 233 251 205 169];
x = [163 166 207 248 210];
P = [x(:) y(:)];

%%%%    Snake
Options = struct;
Options.Verbose = true;
Options.Iterations = 300;
Options.Alpha = 0.01;
Options.Beta = 0.01;
Options.Delta = 0.01;
Options.Kappa = 10;
[O,J] = Snake2D( I , P , Options );

%%%%    ��ʾ���
Irgb(:,:,1)=I;
Irgb(:,:,2)=I;
Irgb(:,:,3)=J;
figure, imshow(Irgb,[]);
hold on; plot([O(:,2);O(1,2)],[O(:,1);O(1,1)]);
