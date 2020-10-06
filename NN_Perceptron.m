 


%%================== Step 0: Generating halfmoon data =====================
%clear;
rad    = 10;   % radius of the half moon
width  = 6;    % width of the half moon
dist   = 0;   % distance between two half moons
num_tr = 1000; % number of training sets
num_te = 2000;  % number of testing sets
num_samp = num_tr+num_te;% number of samples
epochs = 50;
fprintf('Perceptron for Classification\n');
fprintf('_________________________________________\n');
fprintf('Generating halfmoon data ...\n');
fprintf('  ------------------------------------\n');
fprintf('  Points generated: %d\n',num_samp);
fprintf('  Halfmoon radius : %2.1f\n',rad);
fprintf('  Halfmoon width  : %2.1f\n',width);
fprintf('      Distance    : %2.1f\n',dist);
fprintf('  Number of epochs: %d\n',epochs);
fprintf('  ------------------------------------\n');
[data, data_shuffled] = halfmoon(rad,width,dist,num_samp);


num_in = 2;    % number of input neuron
b    = dist/2;  % bias
err    = 0;    
%eta  = 0.95; % learning rate
eta = rand()/2;
w    = [b;zeros(num_in,1)];% initial weights


%% Step 2,3: activation and actual response
st = cputime;
fprintf('Training the perceptron using LMS ...\n');
fprintf('  ------------------------------------\n');
for epoch = 1:epochs,
    shuffle_seq = randperm(num_tr);
    data_shuffled_tr = data_shuffled(:,shuffle_seq);
    for i = 1:num_tr,
        x = [1 ; data_shuffled_tr(1:2,i)]; % fetching data from database
        d = data_shuffled_tr(3,i);         % fetching desired response from database
        y = mysign(w'*x);
        ee(i) = d-y;
        %% Step 4: update of weight
        w_new = w + eta*(d-y)*x;
        
        w = w_new;
    end
    mse(epoch) = mean(ee.^2);
end
fprintf('  Points trained : %d\n',num_tr);
fprintf('       Time cost : %4.2f seconds\n',cputime - st);
fprintf('  ------------------------------------\n');


figure;
plot(mse,'k');
title('Learning curve');
xlabel('Number of epochs');ylabel('MSE');

%%================= Colormaping the figure here ===========================

figure;
hold on;
xmin = min(data_shuffled(1,:));
xmax = max(data_shuffled(1,:));
ymin = min(data_shuffled(2,:));
ymax = max(data_shuffled(2,:));
[x_b,y_b]= meshgrid(xmin:(xmax-xmin)/100:xmax,ymin:(ymax-ymin)/100:ymax);
z_b  = 0*ones(size(x_b));



for x1 = 1 : size(x_b,1)
    for y1 = 1 : size(x_b,2)
        input = [1; x_b(x1,y1); y_b(x1,y1)];
        z_b(x1,y1) = w'*input;
    end
    
end
%% Adding colormap to the final figure
%figure;
sp = pcolor(x_b,y_b,z_b);
% load red_black_colmap;
% %colormap(myhot_light);
% colormap(red_black);
shading flat;

%%============================== Testing ==================================
fprintf('Testing the perceptron using LMS ...\n');
for i = 1 : num_te,
%for i = num_tr+1:num_samp,
    x = [1 ; data_shuffled(1:2,i+num_tr)]; % fetching data for testing
    y(i) = mysign(w'*x);
    %y(i) = w'*x;
    if y(i) == 1 ,
        plot(x(2),x(3),'rx');
    end
    if y(i) == -1,
        plot(x(2),x(3),'k+');
    end
end
xlabel('x');ylabel('y');
title(['Classification using Perceptron with dist = ',num2str(dist), ...
       ', radius = ', num2str(rad), ' and width = ',num2str(width)]);
fprintf('Mission accomplished!\n');
% Calculate testing error rate
for i = 1:num_te,
    if abs(y(i) - data_shuffled(3,i+num_tr)) > 1E-6,
        err = err + 1;
    end
end
fprintf('  ------------------------------------\n');
fprintf('   Points tested : %d\n',num_te);
fprintf('    Error points : %d (%5.2f%%)\n',err,(err/num_te)*100);
fprintf('  ------------------------------------\n');
fprintf('_________________________________________\n');

%%======================= Plot decision boundary ==========================
set(gca,'XLim',[xmin xmax],'YLim',[ymin ymax]);
%% Adding contour to show the boundary
contour(x_b,y_b,z_b,[0 0],'k','Linewidth',1);
%contour(x_b,y_b,z_b,[-1 -1],'k:','Linewidth',2);
%contour(x_b,y_b,z_b,[1 1],'k:','Linewidth',2);
set(gca,'XLim',[xmin xmax],'YLim',[ymin ymax]);
grid on;

