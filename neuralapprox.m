%By B Pradhan, CDS, Indian Institute of Science

%Code from my class project, a neural network for 1D function approximation
%Based on the paper by Cybenko, 1989

%Tentative work, let me know of suboptimalities in code. 

%Single Hidden layer network (Can also be represented as a sum of sigmoids)
%Adaptive learning rate used. The learning rate is different for each
%weight and bias.

clc;clear;

%Function that has to be interpolated
%F = @(x) ( 1./(1 + 25*x.^2) ); 
F = @(x) ( sin(6*pi*x)); 

MAX = 1000; %Number of iterations to train the network

N = 50; %Number of nodes in hidden layer, i.e. number of terms in our series

etamax = 1;


%Learning rate vectors for all weights initialzed
etaW = 0.01*ones(N,1);
etaB = 0.01*ones(N,1);
etaV = 0.01*ones(N,1); 

%Initilizing the weights from a normal distribution

%Be careful about weight initialization
W = randn(N,1); %Weights from the input for the hidden layer, i.e. the w_i's in our series \sum v_i sigmoid(w_i x + b_i)
B = 10*randn(N,1); %Bias at each node at the hidden layer, i.e. the b_i's in our series \sum v_i sigmoid(w_i x + b_i)

V = randn(N,1); %Weights from hidden to output layer, i.e. constants v_i in our series

W = W + abs(min(W))*1.5;

%V = V + abs(min(V))*1.5;
%B = B + abs(min(B))*1.5;

%Intializing variables to store weight values from previous iterations.
W_p = 0*W;
W_pp = W_p;

dW = W_p;

V_p = 0*V;
V_pp = V_p;

dV = V_p;

B_p = 0*B;
B_pp = B_p;

dB = B_p;

%Increase the training size as you increase number of parameters
x = 0:0.001:1; %Training input

Y = F(x);

Tsize = numel(x); %Training input size
O = zeros(1,Tsize); %Initializing Output Array

%Standardizing data
x2 =  (x - mean(x) ) /std(x);  

Y2 =  (Y - mean(Y) ) /std(Y);

WB = waitbar(0,'Please wait...');

figure(1)
for iter = 1:MAX

waitbar(iter/MAX); 
  
H = sigm(W*x2  +  B); %MATLAB adds vector B to every column of W*x

O = V.*(H); %Multiply V elementwise to every column of H. 

O = sum(O,1); %Summing all the terms of the series

%Gradient descent using Backpropagation
%computation
%Quadratic cost function
%C(W,V,B) = \sum_x (1/2n) * ||y - o||^2, o - output node result 

dW_p = dW;

dV_p = dV;

dB_p = dB;

delV = (O - Y2);

dV = H.*delV; 

dV = sum(dV,2)/(Tsize);

delW = H.*(1 - H).*(delV.*V);

dW = x2.*delW;

dW = sum(dW,2)/(Tsize);

dB = delW;

dB = sum(dB,2)/(Tsize);

%Adaptive learning rate
if(iter>4)

etaW = abs((W_p - W_pp)./(dW - dW_p));
etaW(etaW > etamax) = etamax;


etaV = abs( (V_p - V_pp)./(dV - dV_p) );
etaV(etaV > etamax) = etamax;


etaB = abs((B_p - B_pp)./(dB - dB_p) );
etaB(etaB > etamax) = etamax;

end
%Gradient Descent
W = W - etaW.*dW;
V = V - etaV.*dV;
B = B - etaB.*dB;

W_pp = W_p;
W_p = W;

V_pp = V_p;
V_p = V;

B_pp = B_p;
B_p = B;

%E_RR stores mean square error for every iteration
E_RR(iter) = sum((O - Y2).^2)/Tsize; 
plot(x2,O,x2,Y2, 'linewidth',2.5);
end

close(WB);

x3 = 0:0.0033333:1;

x4 =  (x3 - mean(x3) ) /std(x3);  

Y3 = F(x3);

H = sigm(W*x4  +  B);
O = V.*(H);
O = sum(O,1);

%Rescaling the output. Remember we trained our neural netowork on
%standardized data

O = O*std(Y3) + mean(Y3); 

err = sum((O - Y3).^2); 
err = sqrt(err)/numel(Y3);


%Sigmoid function
function y = sigm(x)

y = 1./(1 + exp(-x));

end
