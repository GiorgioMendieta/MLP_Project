%% Proyecto final - Jorge Mendieta
clear all;
clc;

% Half moon parameters
rad     = 12;
width   = 5;
dist    = -6;
nLearn  = 1000;  % At least 100 samples required to have a precision of 1% 
nTest   = 2000;    
nSamp   = nLearn + nTest;

[datos, datosRev]   = halfmoon(rad,width,dist,nSamp);

% Neural network parameters
nIn     = 2;   % Input  layer neurons
nHid    = 20;  % Hidden layer neurons
nOut    = 1;   % Output layer neurons

% Normal distribution initialization of weight vector
w1{1}   =  rand(nHid, nIn + 1); % Initialize weights of input layer to hidden layer
dw0{1}  = zeros(nHid, nIn + 1); % Initialize vector
w1{2}   =  rand(nOut,nHid + 1); % Initialize weights of hidden layer to output layer
dw0{2}  = zeros(nOut,nHid + 1); % Initialize vector

% Hyperparameters
etaHid  = 0.2;
etaOut  = etaHid; % Same learning rate for all neurons
alpha   = 0.4;  % Momentum constant
nEpochs = 50;

% Misc. variables
nErrors = 0;    % Error counter

fprintf('Proyecto Final - Jorge Mendieta\n\n');
fprintf('Multilayer Perceptron (MLP)\n');
fprintf('Using hyperbolic tangent activation function\n');
fprintf('Learning samples: N = %i\n', nLearn);
fprintf('Testing samples : M = %i\n', nTest);
fprintf('------------------------------------\n');

%% Plot of the half moons

figure(1);
set(gcf,'Position',[1000 100 500 800])
subplot(2,1,1);
hold on; grid on;
axis tight; axis equal;
title(['Learning: r = ' num2str(rad) ', w = ' num2str(width) ', d = ' num2str(dist) ', N = ' num2str(nLearn) ' samples']);
% Outer circumference
xC11  = -(rad+width/2):0.01:(rad+width/2);
yC11  = sqrt((rad+width/2)^2 - xC11.^2);
% Inner circumference
xC12  = -(rad-width/2):0.01:(rad-width/2);
yC12  = sqrt((rad-width/2)^2 - xC12.^2);
% First half-moon
plot(xC11,yC11,'r');
plot(xC12,yC12,'r');
plot([-(rad+width/2) -(rad-width/2)], [0 0], 'r');
plot([(rad+width/2) (rad-width/2)], [0 0], 'r');
% Second half-moon
plot((xC11+rad),(-yC11-dist),'b');
plot((xC12+rad),(-yC12-dist),'b');
plot([-(width/2) (width/2)], [-dist -dist], 'b');
plot([(2*rad-width/2) (2*rad+width/2)], [-dist -dist], 'b');

% Plot clusters
for i = 1:nLearn
    % Desired response
    if (datosRev(3,i) == 1)
        % x coordinate, y coordinate
        plot(datosRev(1,i),datosRev(2,i),'r.');
    else
        plot(datosRev(1,i),datosRev(2,i),'b.');
    end  
end


%% ---------------------------- Learning stage ---------------------------- %

% Preprocesamiento de datos
mean1 = [mean(datos(1:2,:)')';0];
for i = 1:nSamp,
    nor_data(:,i) = datosRev(:,i) - mean1;
end
max1 = [max(abs(nor_data(1:2,:)'))';1];
for i = 1:nSamp,
    nor_data(:,i) = nor_data(:,i)./max1;
end

fprintf('Beginning training stage ...\n');

for epoch = 1:nEpochs
    shuffleSeq  = randperm(nLearn);        % Perform a random permutation of data
    nor_dataL   = nor_data(:, shuffleSeq); % Shuffle the data to learn
    
    for i = 1:nLearn       
        % Fetch data
        x       = [nor_dataL(1:2, i); 1];  % Input vector 3x1 = x coord, y coord, Fixed input
        d       =  nor_dataL(3, i);        % Desired response
        
        % Forward pass computation
        vHid    = w1{1}*x;
        hidN    = [activation(vHid);1];      % Non-linear hidden neurons      
        vOut    = w1{2}*hidN;             
        outN    =  activation(vOut);         % Non-linear output neuron
        
        error(:,i)= d - outN;
        
        % Backward pass computation using local gradients
        deltaOut= error(:,i).*d_activation(vOut);
        deltaHid= d_activation(vHid).*(deltaOut*w1{2}(:,1:nHid)');
       
        % Weight correction
        dw1{1}  = etaHid*deltaHid*x';
        dw1{2}  = etaOut*deltaOut*hidN';
        
        % Update of weight vectors
        w2{1}   = w1{1} + alpha*dw0{1} + dw1{1};  % input  weights -> hidden
        w2{2}   = w1{2} + alpha*dw0{2} + dw1{2};  % hidden weights -> output
        
        dw0     = dw1;
        w1      = w2;
    end
    
    % Loss function with MSE
    loss(epoch) = mean(error'.^2);
end
fprintf('Training stage completed!\n');
%fprintf('Final weights : w = %.5f\n', w);
fprintf('------------------------------------\n');

% Plot learning curve
figure(2);
hold on; grid on;
plot(loss,'r- .');
ax = gca;
ax.XAxis.TickLabelFormat = '%d';
title(['Loss / MSE']);
xlabel('Epoch #');ylabel('MSE');


%% Plot of the half moons
% Plotting the clusters
figure(1);
subplot(2,1,2);
hold on; grid on;
axis tight; axis equal;
% First half-moon
% plot(xC11,yC11,'r');
% plot(xC12,yC12,'r');
% plot([-(rad+width/2) -(rad-width/2)], [0 0], 'r');
% plot([(rad+width/2)   (rad-width/2)], [0 0], 'r');
% Second half-moon
% plot((xC11+rad),(-yC11-dist),'b');
% plot((xC12+rad),(-yC12-dist),'b');
% plot([-(width/2) (width/2)], [-dist -dist], 'b');
% plot([(2*rad-width/2) (2*rad+width/2)], [-dist -dist], 'b');


%% ----------------------------- Testing stage -----------------------------%
fprintf('Beginning testing stage ...\n');
 
for i = nLearn+1:nSamp
    x = [nor_data(1:2, i); 1]; % Input vector 3x1 = x coord, y coord, Fixed input

    hidN        = [activation(w1{1}*x);1];
    outN(:,i)   =  activation(w1{2}*hidN);

    % Plot clusters
    if outN(:,i)>0
        % x coordinate, y coordinate
        plot(x(1),x(2),'r.');
    end
    if outN(:,i)<0
        plot(x(1),x(2),'b.');
    end  

    y = signum(outN(i)) - nor_data(3,i);
    % Check for errors
    if abs(y) > 1E-6,
        nErrors = nErrors + 1;
    end
end 

fprintf('Testing stage completed!\n');
fprintf('------------------------------------\n');
%Percentage of errors
errorPerc =  nErrors*100/nTest;
fprintf('Error points : %d (%5.2f%%)\n', nErrors, errorPerc);

title(['Test: M = ' num2str(nTest) ' samples, Error = ', num2str(errorPerc) '%']);