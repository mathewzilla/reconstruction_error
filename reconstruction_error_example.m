%% Williams style reconstruction error of random matrix.
clear all
rng(15); % Fix random number generator
pausing = 0; % Pause when plotting?
% Start with data matrix (e.g. T x N, but random here)
O = 100;                              % Observations e.g. time points
F = 50;                               % Features     e.g. neuron firing rates

X = randn(O,F);
X = X - mean(X); % Subtract column means. NB if we normalize (z-score) we 
% get exact same answer as corrcoef(X). i.e. cov(zscore(X)) = corrcoef(X)

C = cov(X);      
[V,D] = eig(C);
W = X*V;         % Weights or 'loadings'

clear bigX bigX_lowD
clear rec_error_highD
figure(6); clf
for i = 1:F
    % Reconstruction based on increasing numbers of eigenvectors
    
    % NB eigenvalues are in ascending order by default, so taking
    % eigenvalues from the end and working backwards...
    X_r = W(:,(F+1-i):end)*V(:,(F+1-i):end)';
    X_error = X - X_r;
    big_X(i,:,:) = X_r; % Save error matrices for later comparison
    
    rec_error_highD(i) = sqrt(sum(X_error(:).^2)); % reconstruction error
    
    if i < 11
        subplot(1,3,1)
        imagesc(X); axis image; title X; caxis([-3 3]);
        subplot(1,3,2)
        imagesc(X_r); axis image; title X_r; caxis([-3 3]);
        subplot(1,3,3)
        imagesc(X_error); axis image; title('X - X_r'); caxis([-3 3]);
        suptitle(['Eig = ',num2str(i)])
        drawnow; 
        if pausing
            pause
        end
    end
    
    
end

% repeat but with 5D structure
X_lowD = [];
for i = 1:5
    X_lowD = [X_lowD,ones(O,10).*randn(O,1)];
end
X_lowD = X_lowD - mean(X_lowD);
C_lowD = cov(X_lowD);
[V_lowD,D_lowD] = eig(C_lowD);
W_lowD = X_lowD*V_lowD;

clear rec_error_lowD
figure(7); clf
for i = 1:F
    X_r = W_lowD(:,(F+1-i):end)*V_lowD(:,(F+1-i):end)';
    X_error = X_lowD - X_r;
    big_X_lowD(i,:,:) = X_r;
    rec_error_lowD(i) = sqrt(sum(X_error(:).^2)); % Or norm(X_error,'fro')
    if i < 11
        subplot(1,3,1)
        imagesc(X_lowD); axis image; title X_lowD; caxis([-3 3]);
        subplot(1,3,2)
        imagesc(X_r); axis image; title X_r; caxis([-3 3]);
        subplot(1,3,3)
        imagesc(X_error); axis image; title('X - X_r'); caxis([-3 3]);
        suptitle(['Eig = ',num2str(i)])
        drawnow;   
        if pausing
            pause;
        end
    end
end

figure(8);
clf
plot(rec_error_highD);
hold all
plot(rec_error_lowD);
ylabel('Reconstruction error')
xlabel('Eigenvectors')
legend('High D','Low (5) D')
