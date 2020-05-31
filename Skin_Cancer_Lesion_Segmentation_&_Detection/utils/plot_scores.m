function [ ] = plot_scores( )
%ACC_PLOT plots Loss and Accuracy vs Epoch 

train_acc = csvread('train_acc.txt');
train_loss = csvread('train_loss.txt');

val_acc = csvread('val_acc.txt');
val_loss = csvread('val_loss.txt');

epochs = length(train_acc);

%Plot accuracy
x = linspace(0,epochs,epochs);
figure
plot(x,train_acc,'DisplayName','Training')
hold on
plot(x,val_acc, 'DisplayName','Validation')

title('Accuracy')
xlabel('Epochs')
ylabel('Normalized Accuracy')
legend('show')

%Plot Loss
figure
plot(x,train_loss,'DisplayName','Training')
hold on
plot(x,val_loss, 'DisplayName','Validation')

title('Loss')
xlabel('Epochs')
ylabel('Loss (Binary Cross Entropy)')
legend('show')


% Compute best validation model Loss and Accuracy
best_acc = max(val_acc);
best_loss = min(val_loss);

fprintf('Validation Accuracy: %f \n', best_acc);
fprintf('Validation Loss: %f \n', best_loss);

end
