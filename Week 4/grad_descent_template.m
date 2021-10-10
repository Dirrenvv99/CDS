clear all               % always good idea to start with nothing, to avoid errors
rand('seed',0);         % fixed seeds allow you to debug your code better, should be removed later to test the rebustness of your results.
randn('seed',0);

METHOD='newton';
%METHOD='gd';
METHOD='sgd';
%METHOD='line search';

% DATA
load /Users/bertkappen/doc/ml_murphy/pmtk3-master/pmtkdataCopy/mnistAll.mat;
X=double(mnist.train_images);
y=mnist.train_labels;

X3=X(:,:,find(y==3));
X7=X(:,:,find(y==7));
n3=size(X3,3);
n7=size(X7,3);
X3=reshape(X3,784,size(X3,3));
X7=reshape(X7,784,size(X7,3));
X=[X3,X7];                      % training set input
t=[zeros(1,n3),ones(1,n7)];     % training set output
n=size(X,1);
N=size(X,2);
X=X/max(max(X));

X(n+1,:)=1;                     % add x=1 to encode bias in w(n+1)
n=n+1;

X1=double(mnist.test_images);
y1=mnist.test_labels;
    
X3=X1(:,:,find(y1==3));
X7=X1(:,:,find(y1==7));
n3=size(X3,3);
n7=size(X7,3);
X3=reshape(X3,784,size(X3,3));
X7=reshape(X7,784,size(X7,3));
X1=[X3,X7];                      % test set input
t1=[zeros(1,n3),ones(1,n7)];     % test set output
N1=size(X1,2);
X1=X1/max(max(X1));
X1(n,:)=1;                     % add x=1 to encode bias in w(n+1)

% END DATA

% insert your code here

%introduce tic, toc to measure cpu time
%tic
%your code
%toc

