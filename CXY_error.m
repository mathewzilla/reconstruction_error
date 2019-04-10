% CXY_error.m
%
% Script for working through PCA and reconstruction error, and possible
% equivalence in measuring CXY reconstruction error in spectral noise
% rejection

% The wiki on PCA describes the rows and columns (and the purpose of PCA)
% in clearer terms than Williams' post - "Suppose you have data comprising
% a set of observations of p variables, and you want to reduce the data so
% that each observation can be described with only L variables, L < p.
% Suppose further, that the data are arranged as a set of n data vectors
% x_1 ... x_n with each x_i representing a single grouped observation of
% the p variables."

% I.E. each time point is an observation, and each neuron's firing is a variable
% at each observation. Therefore the N*T matrix needs to be a T*N matrix
% (like Mark has it in the Gdoc, but not like Williams has it in his
% introduction).

run figure_properties

%% Show in a simple case that PCA recovers latent structure
rng(13); % Fix random number generator
% Start with data matrix (e.g. N x T, but random here)
O = 100;                              % Observations e.g. time points
F = 50;                               % Features     e.g. neuron firing rates

data = 0.2*randn(O,F);                % O*F or T*N, I*J in Williams' blog

signal = smooth(randn(1,F))';         % Some latent structure i.e. a component
weight = smooth((randn(O,1)));        % A weight for each O/N i.e. a loading
data2 = (data + (signal.*weight));    % Add latent structure to random matrix
data2  = (data2' - mean(data2'))';    % Subtract mean from each row
C = cov(data2);                       % Covariance.

figure(1); clf
subplot(2,3,[1,4]);
imagesc(data2); axis image; title('data')
subplot(2,3,3);
imagesc(C); axis image; title('C')

[V,D] = eig(C);
subplot(2,3,6);
imagesc(D); axis image; title('D (eigenvalues on diag)')

proj_E1 = V(:,end);
[proj_E1_sorted,E1_ix] = sort(proj_E1);

subplot(2,3,[2,5]);
imagesc(V(:,end)'.* sum(V(:,end).*data2')'); axis image; title('1 Eig reconstruction')

figure(2); clf
subplot(1,3,1); imagesc(data2); axis image; title('data')
subplot(1,3,2); imagesc(data2(:,E1_ix)); axis image; title('ordered data')
subplot(1,3,3); imagesc(proj_E1_sorted'.* sum(proj_E1_sorted .* data2(:,E1_ix)')'); axis image; title('ordered data (1 Eig recon)')

%% W loadings
for i = 1: F
    W(:,i) = sum(V(:,i) .* data2')';
end

figure(3);
subplot(1,2,1); imagesc(data2); axis image; title data
subplot(1,2,2); imagesc(W*V'); axis image; title ('W*V^{T}')

%% Can also get W loadings as it's done in matlab pca code
W2 = data2/V';
% data*V also works.

%% Progressively better reconstructions
figure(4);
% d = diag(D);
% subplot(1,5,1);
% WCt = d(end)* V(:,end)'.* sum(V(:,end).*data2')';
% imagesc(WCt); axis image; title('1 Eig')
% for i = 1:4
%     WCt = WCt + d(end-i) * V(:,end-i)'.* sum(V(:,end-i).*data2')';
%     subplot(1,5,i+1)
%     imagesc(WCt); axis image; title([num2str(i+1),' Eig'])
% end
%
% for i = 5:49
%     WCt = WCt + d(end-i) * V(:,end-i)'.* sum(V(:,end-i).*data2')';
% end

d = diag(D);
subplot(1,5,1);
WCt = W(:,end) * V(:,end)';
imagesc(WCt); axis image; title('1 Eig')
for i = 1:4
    WCt = WCt + W(:,end-i) * V(:,end-i)';
    subplot(1,5,i+1)
    imagesc(WCt); axis image; title([num2str(i+1),' Eig'])
end

for i = 5:49
    WCt = WCt + W(:,end-i) * V(:,end-i)';
end


figure(5);
subplot(1,2,1); imagesc(data2); axis image; title data
subplot(1,2,2); imagesc(WCt); axis image; title ('reconstructed data (sum ind PCs)')

%% PCA version - requires rotating the matrix to get the behaviour we want?
[coeff,score,latent] = pca(data2);

%% In PCA land the 'coeff' is the principal component. 'score' is the loading on

figure(3); clf
% coeff(:,1) recovers the latent signal (principal component), which is the
% same as the eigenvector associated with the largest eigenvalue of the
% covariance matrix C
subplot(2,1,1)
plot(zscore(signal))
hold all
plot(zscore(-coeff(:,1)))
plot(zscore(V(:,end)))
legend('signal','coeff(:,1)','V(:,end)')

% score(:,1) recovers the weight or loading, which is the same as the
% summed projection of the eigenvector associated with the largest
% eigenvalue onto the data space (V * data).
subplot(2,1,2)
plot(zscore(weight))
hold all
plot(-zscore(score(:,1)))
plot(zscore(sum(V(:,end).*data2')))
legend('weight','score(:,1)','sum(V(:,end).*X)')

%% Image plot of 1PC reconstruction of data, with pca and eig
figure(6);
subplot(1,3,1)
imagesc(data2); title data
subplot(1,3,2);
imagesc(coeff(:,1)'.*score(:,1)); title pca
subplot(1,3,3);
imagesc(V(:,end)'.* sum(V(:,end).*data2')'); title eig

%% Williams style reconstruction error of random matrix.
clear all
rng(15); % Fix random number generator
pausing = 0; % Pause when plotting?
% Start with data matrix (e.g. T x N, but random here)
O = 100;                              % Observations e.g. time points
F = 50;                               % Features     e.g. neuron firing rates

X = randn(O,F);
X = X - mean(X); % Subtract column means.
% NB if we normalize (z-score) we get exact same answer as corrcoef(X);
% i.g. cov(zscore(X)) = corrcoef(X)

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

%% Reconstructing C instead of X

% reconstruct the "signal" correlation matrix
bigC = [];
Cs = zeros(F);
for i= 1:F
    Cs = Cs + D(F+1-i,F+1-i) * V(:,F+1-i)*V(:,F+1-i)';
    bigC(i,:,:) = Cs;
end

% Same for low D version
% reconstruct the "signal" correlation matrix
bigC_lowD = [];
Cs_lowD = zeros(F);
for i= 1:F
    Cs_lowD = Cs_lowD + D_lowD(F+1-i,F+1-i) * V_lowD(:,F+1-i)*V_lowD(:,F+1-i)';
    bigC_lowD(i,:,:) = Cs_lowD;
end

%% loop displaying the first 10 reconstructions
clf
subplot(2,2,1);
imagesc(C); axis image;
subplot(2,2,3);
imagesc(C_lowD); axis image;

for i = 1:10
    subplot(2,2,2)
    imagesc(squeeze(bigC(i,:,:)));
    axis image;
    subplot(2,2,4)
    imagesc(squeeze(bigC_lowD(i,:,:)));
    axis image;
    %     suptitle(['Eig ',num2str(i)]);
    drawnow
    %     pause;
end

%% compute reconstruction error for C, then compare to same for X

for i = 1:F
    C_error(i) = norm(C - squeeze(bigC(i,:,:)),'fro');
    C_error_lowD(i) = norm(C_lowD - squeeze(bigC_lowD(i,:,:)),'fro');
end

figure(7);
plot(C_error)
plot(C_error_lowD);
legend('X_highD_error','X_lowD_error','C_highD_error','C_lowD_error')

%% better comparison of X and C error
figure (9); clf
plot(C_error,rec_error_highD,'.')
hold all
plot(C_error_lowD,rec_error_lowD,'.')

%% Corrcoef between X and bigX, C and bigC
for i = 1:F
    PCC = corrcoef(X,big_X(i,:,:)); pcc_X(i) = PCC(2,1);
    PCC = corrcoef(X_lowD,big_X_lowD(i,:,:)); pcc_X_lowD(i) = PCC(2,1);
    PCC = corrcoef(C,bigC(i,:,:)); pcc_C(i) = PCC(2,1);
    PCC = corrcoef(C_lowD,bigC_lowD(i,:,:)); pcc_C_lowD(i) = PCC(2,1);
end
figure(10); clf;
plot(pcc_X); hold all
plot(pcc_X_lowD)
plot(pcc_C)
plot(pcc_C_lowD)
legend('X_{pcc}','X_{lowD}_{pcc}','C_{pcc}','C_{lowD}_{pcc}')

%% A grid of datasets, higher dim, greater length,
figure(11); clf;
rng(13);
lengths = linspace(10,200,20);
dims = linspace(1,20,20);
F = 50;
clear M_error_fro D_error_fro M_error_pcc D_error_pcc
for l = 1:numel(lengths)
    O = lengths(l);
    for d = 1:numel(dims)
        X = [];
        for i = 1:dims(d)
            X = [X,ones(O,floor(F/dims(d))).*randn(O,1)];
        end
        X = [X,0.1*randn(O,F-size(X,2))];
        X = (X - mean(X))';
        C = cov(X);
        [V,D] = eig(C);
        % Flip around
        D = D(linspace(F,1,F),linspace(F,1,F));
        V = V(:,linspace(F,1,F));
        W = X*V;
        % loop reconstructing C and X, and computing D_error and M_error
        Cs = zeros(F);
        Xs = zeros(F);
        
        for i = 1:F
            Cs = Cs + D(i,i) * V(:,i)*V(:,i)';
            Xs = W(:,1:i)*V(:,1:i)';
            M_error_fro(l,d,i) = norm(C-Cs,'fro');
            D_error_fro(l,d,i) = norm(X-Xs,'fro');
            %             if l == 5 % Don't want to look at everything
            %                 if i<10
            %                     subplot(2,3,[1,4])
            %                     imagesc(X); axis image
            %                     subplot(2,3,[2,5])
            %                     imagesc(Xs); axis image
            %                     subplot(2,3,3)
            %                     imagesc(C); axis image
            %                     subplot(2,3,6)
            %                     imagesc(Cs); axis image
            %                     drawnow;
            %                 end
            %             end
            PCC_M = corrcoef(C,Cs);
            PCC_D = corrcoef(X,Xs);
            
            M_error_pcc(l,d,i) = PCC_M(2,1);
            D_error_pcc(l,d,i) = PCC_D(2,1);
        end
        
    end
end

%% Compare metrics
% figure(12); clf
figure('Units', 'centimeters', 'PaperPositionMode', 'auto','Position',[10 15 figsize.retina]);

cmap = varycolor(20);

for i = 1:20
    subplot(2,2,1);
    x= M_error_fro(i,:,:);
    y= D_error_fro(i,:,:);
    % plot(M_error_fro(i,:,:),D_error_fro(:),'.')
    plot(x(:),y(:),'.','color',cmap(i,:)); hold all; axis square
    xlabel('M_{error} (Frob)')
    ylabel('D_{error} (Frob)')
    PCC = corrcoef(M_error_fro(:),D_error_fro(:));
    title(['Frob. PCC = ',num2str(PCC(2,1))])
    
    subplot(2,2,2);
    x= M_error_pcc(i,:,:);
    y= D_error_pcc(i,:,:);
    % plot(M_error_pcc(i,:,:),D_error_pcc(:),'.')
    plot(x(:),y(:),'.','color',cmap(i,:)); hold all; axis square
    xlabel('M_{error} (PCC)')
    ylabel('D_{error} (PCC)')
    PCC = corrcoef(M_error_pcc(:),D_error_pcc(:));
    title(['PCC. PCC = ',num2str(PCC(2,1))])
    
    subplot(2,2,3);
    x= M_error_fro(:,i,:);
    y= D_error_fro(:,i,:);
    % plot(M_error_fro(i,:,:),D_error_fro(:),'.')
    plot(x(:),y(:),'.','color',cmap(i,:)); hold all; axis square
    xlabel('M_{error} (Frob)')
    ylabel('D_{error} (Frob)')
    PCC = corrcoef(M_error_fro(:),D_error_fro(:));
    title(['Frob. PCC = ',num2str(PCC(2,1))])
    
    subplot(2,2,4);
    x= M_error_pcc(:,i,:);
    y= D_error_pcc(:,i,:);
    % plot(M_error_pcc(i,:,:),D_error_pcc(:),'.')
    plot(x(:),y(:),'.','color',cmap(i,:)); hold all; axis square
    xlabel('M_{error} (PCC)')
    ylabel('D_{error} (PCC)')
    PCC = corrcoef(M_error_pcc(:),D_error_pcc(:));
    title(['PCC. PCC = ',num2str(PCC(2,1))])
    drawnow
end

FormatFig_For_Export(gcf,fontsize,fontname,widths.axis);
print([exportpath 'D_error_vs_M_error'],'-dpdf');



%% For square dataset, plot data as lines (lengths(5))
figure('Units', 'centimeters', 'PaperPositionMode', 'auto','Position',[10 15 [10,15]]);

cmap = varycolor(20);
for i = 1:20
    
    subplot(3,2,1);
    x= M_error_fro(5,i,:);
    y= D_error_fro(5,i,:);
    % plot(M_error_fro(i,:,:),D_error_fro(:),'.')
    plot(x(:),y(:),'.','color',cmap(i,:)); hold all; axis square
    xlabel('M_{error} (Frob)')
    ylabel('D_{error} (Frob)')
    PCC = corrcoef(x,y);
    title(['Frob. PCC = ',num2str(PCC(2,1))])
    
    subplot(3,2,2);
    x= M_error_pcc(5,i,:);
    y= D_error_pcc(5,i,:);
    % plot(M_error_pcc(i,:,:),D_error_pcc(:),'.')
    plot(x(:),y(:),'.','color',cmap(i,:)); hold all; axis square
    xlabel('M_{error} (PCC)')
    ylabel('D_{error} (PCC)')
    PCC = corrcoef(x,y);
    title(['PCC. PCC = ',num2str(PCC(2,1))])
    
    
    subplot(3,2,3);
    x = squeeze(M_error_fro(5,i,:));
    y = squeeze(D_error_fro(5,i,:));
    plot(x,'color',cmap(i,:)); hold all; axis square
    xlabel('Eigs')
    ylabel('M_{error} (Frob)')
    
    subplot(3,2,5);
    plot(y,'color',cmap(i,:)); hold all; axis square
    xlabel('Eigs')
    ylabel('D_{error} (Frob)')
    
    subplot(3,2,4);
    x = squeeze(M_error_pcc(5,i,:));
    y = squeeze(D_error_pcc(5,i,:));
    plot(x,'color',cmap(i,:)); hold all; axis square
    xlabel('Eigs')
    ylabel('M_{error} (PCC)')
    
    subplot(3,2,6);
    plot(y,'color',cmap(i,:)); hold all; axis square
    xlabel('Eigs')
    ylabel('D_{error} (PCC)')
    
    drawnow
end

suptitle('Colour = dims (1 - 20). O = 50, F = 50.')

FormatFig_For_Export(gcf,fontsize,fontname,widths.axis);
print([exportpath 'D_error_vs_M_error_square_lines'],'-dpdf');

%% Spectral rejection M_error
rng(13);
O = 100;
F = 50;
d = 4;
X = [];
for i = 1:d
    X = [X,ones(O,10).*randn(O,1)];
end
% Add column of zeros, then noise over the top
X = [X, zeros(O,F-size(X,2))];
X = X + 0.1*randn(O,F);

X = X - mean(X);

Data = reject_the_noise(X);

% Get eigs of C for comparison
C = cov(X);
[V,D] = eig(C);
% Flip around
D = D(linspace(F,1,F),linspace(F,1,F));
V = V(:,linspace(F,1,F));
W = X*V;

% Now get eigs of B
B = Data.A - Data.ExpA;
[VB,DB] = eig(B);

% Flip around
DB = DB(linspace(F,1,F),linspace(F,1,F));
VB = VB(:,linspace(F,1,F));

% Loadings of VB on to the data
WB = X*VB;

%% What about eigs of A and ExpA. Do they sum to create the eigs of B? - No
% [VA,DA] = eig(Data.A);
% [VExpA,DExpA] = eig(Data.ExpA);
%
% WA = Data.A*VA;
% WExpA = Data.ExpA*VExpA;

%% Reconstructing B from V and D.
CsB = zeros(F);
XsB = zeros(F);
Cs = zeros(F);
Xs = zeros(F);
clear MB_error DB_error M_error D_error
for i = 1:F
    CsB = CsB + DB(i,i) * VB(:,i)*VB(:,i)';
    XsB = WB(:,1:i)*VB(:,1:i)';
    
    % Compare to non-spectral rejected version
    Cs = Cs + D(i,i) * V(:,i)*V(:,i)';
    Xs = W(:,1:i)*V(:,1:i)';
    % Error compute
    MB_error(i) = norm(B-CsB,'fro');
    DB_error(i) = norm(X-XsB,'fro');
    
    M_error(i) = norm(C-Cs,'fro');
    D_error(i) = norm(X-Xs,'fro');
end


% Plot both metrics and eigs, with line for spectral rejection dims
figure('Units', 'centimeters', 'PaperPositionMode', 'auto','Position',[10 15 [25,10]]);

% figure(9); clf
subplot(1,3,1)
plot(M_error,'linewidth',2)
hold all
plot(MB_error,'linewidth',2)
plot(Data.Dn*ones(2,1),[0,max([M_error,MB_error])],'k--','linewidth',2);
plot(d*ones(2,1),[0,max([M_error,MB_error])],'-','color',[.5,.5,.5],'linewidth',2);
axis square; xlim([0,10]); legend('M_{error}','MB_{error}','Data.Dn','N clusters')

subplot(1,3,2)
plot(D_error,'linewidth',2)
hold all
plot(DB_error,'linewidth',2)
plot(Data.Dn*ones(2,1),[0,max([D_error,DB_error])],'k--','linewidth',2);
plot(d*ones(2,1),[0,max([D_error,DB_error])],'-','color',[.5,.5,.5],'linewidth',2);
axis square; xlim([0,10]); legend('D_{error}','DB_{error}','Data.Dn','N clusters')


subplot(1,3,3)
plot(diag(D),'linewidth',2)
hold all
plot(diag(DB),'linewidth',2)
plot(Data.Dn*ones(2,1),[0,max([diag(D);diag(DB)])],'k--','linewidth',2);
plot(d*ones(2,1),[0,max([diag(D);diag(DB)])],'-','color',[.5,.5,.5],'linewidth',2);
axis square; xlim([0,10]); legend('Eigs_A','Eigs_B','Data.Dn','N clusters')

FormatFig_For_Export(gcf,fontsize,fontname,widths.axis);
print([exportpath 'NR_5D_1N'],'-dpdf');

%% Same again but added levels of noise - NB this is data noise not
% network noise as in Spectral Rejection paper

%% Reconstructing B from V and D.

clear MB_error DB_error M_error D_error

rng(13);
O = 100;
F = 50;
d = 4;

NL = linspace(0.01,1,10);

clear All_Data
for n = 1:10
    clear C D V W B VB DB
    X = [];
    for i = 1:d
        X = [X,ones(O,10).*randn(O,1)];
    end
    % Add column of zeros, then noise over the top
    X = [X, zeros(O,F-size(X,2))];
    
    % Vary noise level
    X = X + NL(n)*randn(O,F);
    
    X = X - mean(X);
    
    Data = reject_the_noise(X);
    All_Data{n} = Data;
    
    % Get eigs of C for comparison
    C = cov(X);
    [V,D] = eig(C);
    % Flip around
    D = D(linspace(F,1,F),linspace(F,1,F));
    V = V(:,linspace(F,1,F));
    W = X*V;
    
    % Now get eigs of B
    B = Data.A - Data.ExpA;
    [VB,DB] = eig(B);
    
    % Flip around
    DB = DB(linspace(F,1,F),linspace(F,1,F));
    VB = VB(:,linspace(F,1,F));
    
    % Loadings of VB on to the data
    WB = X*VB;
    
    CsB = zeros(F);
    XsB = zeros(F);
    Cs = zeros(F);
    Xs = zeros(F);
    
    for i = 1:F
        CsB = CsB + DB(i,i) * VB(:,i)*VB(:,i)';
        XsB = WB(:,1:i)*VB(:,1:i)';
        
        % Compare to non-spectral rejected version
        Cs = Cs + D(i,i) * V(:,i)*V(:,i)';
        Xs = W(:,1:i)*V(:,1:i)';
        % Error compute
        MB_error(i,n) = norm(B-CsB,'fro');
        DB_error(i,n) = norm(X-XsB,'fro');
        
        M_error(i,n) = norm(C-Cs,'fro');
        D_error(i,n) = norm(X-Xs,'fro');
    end
    
end

save('SR_4D_10reps.mat','MB_error','DB_error','M_error','D_error','All_Data')

%% Forgot to save D and DB for each dataset so regenerating them here
% load('SR_4D_10reps.mat')

rng(13);
O = 100;
F = 50;
d = 4;

NL = linspace(0.01,1,10);

for n = 1:10
    clear C D V W B VB DB
    X = [];
    for i = 1:d
        X = [X,ones(O,10).*randn(O,1)];
    end
    % Add column of zeros, then noise over the top
    X = [X, zeros(O,F-size(X,2))];
    
    % Vary noise level
    X = X + NL(n)*randn(O,F);
    
    X = X - mean(X);
    
    Data = All_Data{n};
    
    % Get eigs of C for comparison
    C = cov(X);
    [V,D] = eig(C);
    % Flip around
    D = D(linspace(F,1,F),linspace(F,1,F));
    V = V(:,linspace(F,1,F));
    W = X*V;
    
    % Now get eigs of B
    B = Data.A - Data.ExpA;
    [VB,DB] = eig(B);
    
    % Flip around
    DB = DB(linspace(F,1,F),linspace(F,1,F));
    VB = VB(:,linspace(F,1,F));
    
    % Loadings of VB on to the data
    WB = X*VB;
    
    Eigy.D = D;
    Eigy.V = V;
    Eigy.DB = DB;
    Eigy.VB = VB;
    Eigy.W = W;
    Eigy.WB = WB;
    
    All_eig{n} = Eigy;
end

save('SR_4D_10reps.mat','MB_error','DB_error','M_error','D_error','All_Data','All_eig')

%% Plot all data
% Plot both metrics and eigs, with line for spectral rejection dims
load('SR_4D_10reps.mat')
figure('Units', 'centimeters', 'PaperPositionMode', 'auto','Position',[10 15 [25,10]]);

d = 4
D_colour = brewermap(10,'*Blues');
B_colour = brewermap(10,'*Oranges');
% figure(10); clf; clear ax
for n = 1:10
    Data = All_Data{n};
    Eigy = All_eig{n};
    ax(1) = subplot(1,3,1);
    plot(M_error(:,n),'color',D_colour(n,:),'linewidth',2)
    hold all
    plot(MB_error(:,n),'color',B_colour(n,:),'linewidth',2)
    plot(Data.Dn*ones(2,1),[0,max([M_error(:);MB_error(:)])],'k--','linewidth',2);
    plot(d*ones(2,1),[0,max([M_error(:);MB_error(:)])],'-','color',[.5,.5,.5],'linewidth',2);
    axis square; xlim([0,10]); legend('M_{error}','MB_{error}','Data.Dn','N clusters');
    
    ax(2) = subplot(1,3,2);
    plot(D_error(:,n),'color',D_colour(n,:),'linewidth',2)
    hold all
    plot(DB_error(:,n),'color',B_colour(n,:),'linewidth',2)
    plot(Data.Dn*ones(2,1),[0,max([D_error(:);DB_error(:)])],'k--','linewidth',2);
    plot(d*ones(2,1),[0,max([D_error(:);DB_error(:)])],'-','color',[.5,.5,.5],'linewidth',2);
    axis square; xlim([0,10]); legend('D_{error}','DB_{error}','Data.Dn','N clusters');
    
    
    ax(3) = subplot(1,3,3);
    plot(diag(Eigy.D),'color',D_colour(n,:),'linewidth',2)
    hold all
    plot(diag(Eigy.DB),'color',B_colour(n,:),'linewidth',2)
    plot(Data.Dn*ones(2,1),[0,max([diag(Eigy.D);diag(Eigy.DB)])],'k--','linewidth',2);
    plot(d*ones(2,1),[0,max([diag(Eigy.D);diag(Eigy.DB)])],'-','color',[.5,.5,.5],'linewidth',2);
    axis square; legend('Eigs_A','Eigs_B','Data.Dn','N clusters');
end
linkaxes(ax,'x'); xlim([0,50]); ylim([0,17])

FormatFig_For_Export(gcf,fontsize,fontname,widths.axis);
print([exportpath 'NR_5D_10N'],'-dpdf');

%% Zoom
xlim([0,10]);
print([exportpath 'NR_5D_10N_zoom'],'-dpdf');

%% More zooming
xlim([2,6]);
print([exportpath 'NR_5D_10N_zoom_zoom'],'-dpdf');

%% 1 level of noise, but increasing d from 1 to 10

clear MB_error DB_error M_error D_error

rng(13);
O = 100;
F = 50;
% d = 4;


clear All_Data All_eig
for n = 1:10
    clear C D V W B VB DB
    X = [];
    d = n;
    for i = 1:d
        X = [X,ones(O,5).*randn(O,1)];
    end
    % Add column of zeros, then noise over the top
    X = [X, zeros(O,F-size(X,2))];
    
    % Vary noise level
    X = X + 0.2*randn(O,F);
    
    X = X - mean(X);
    
    Data = reject_the_noise(X);
    All_Data{n} = Data;
    
    % Get eigs of C for comparison
    C = cov(X);
    [V,D] = eig(C);
    % Flip around
    D = D(linspace(F,1,F),linspace(F,1,F));
    V = V(:,linspace(F,1,F));
    W = X*V;
    
    % Now get eigs of B
    B = Data.A - Data.ExpA;
    [VB,DB] = eig(B);
    
    % Flip around
    DB = DB(linspace(F,1,F),linspace(F,1,F));
    VB = VB(:,linspace(F,1,F));
    
    % Loadings of VB on to the data
    WB = X*VB;
    
    CsB = zeros(F);
    XsB = zeros(F);
    Cs = zeros(F);
    Xs = zeros(F);
    
    for i = 1:F
        CsB = CsB + DB(i,i) * VB(:,i)*VB(:,i)';
        XsB = WB(:,1:i)*VB(:,1:i)';
        
        % Compare to non-spectral rejected version
        Cs = Cs + D(i,i) * V(:,i)*V(:,i)';
        Xs = W(:,1:i)*V(:,1:i)';
        % Error compute
        MB_error(i,n) = norm(B-CsB,'fro');
        DB_error(i,n) = norm(X-XsB,'fro');
        
        M_error(i,n) = norm(C-Cs,'fro');
        D_error(i,n) = norm(X-Xs,'fro');
    end
    
    Eigy.D = D;
    Eigy.V = V;
    Eigy.DB = DB;
    Eigy.VB = VB;
    Eigy.W = W;
    Eigy.WB = WB;
    
    All_eig{n} = Eigy;
    
end

save('SR_10D_1N.mat','MB_error','DB_error','M_error','D_error','All_Data','All_eig')

%% Plot all data
% Plot both metrics and eigs, with line for spectral rejection dims
load('SR_10D_1N.mat')
figure('Units', 'centimeters', 'PaperPositionMode', 'auto','Position',[10 15 [25,10]]);

P_colour = brewermap(12,'Paired');
D_colour = P_colour(1:2:end,:);
B_colour = P_colour(2:2:end,:);
% figure(10); clf; clear ax
for n = 1:5
    Data = All_Data{n+2};
    Eigy = All_eig{n+2};
    ax(1) = subplot(1,3,1);
    plot(M_error(:,n+2),'color',D_colour(n,:),'linewidth',2)
    hold all
    plot(MB_error(:,n+2),'color',B_colour(n,:),'linewidth',2)
    %     plot(Data.Dn*ones(2,1),[0,max([M_error(:);MB_error(:)])],'k--','linewidth',2);
    %     plot(d*ones(2,1),[0,max([M_error(:);MB_error(:)])],'-','color',[.5,.5,.5],'linewidth',2);
    axis square; xlim([0,10]); legend('M_{error}','MB_{error}')
    
    ax(2) = subplot(1,3,2);
    plot(D_error(:,n+2),'color',D_colour(n,:),'linewidth',2)
    hold all
    plot(DB_error(:,n+2),'color',B_colour(n,:),'linewidth',2)
    %     plot(Data.Dn*ones(2,1),[0,max([D_error(:);DB_error(:)])],'k--','linewidth',2);
    %     plot(d*ones(2,1),[0,max([D_error(:);DB_error(:)])],'-','color',[.5,.5,.5],'linewidth',2);
    axis square; xlim([0,10]); legend('D_{error}','DB_{error}')
    
    
    ax(3) = subplot(1,3,3);
    plot(diag(Eigy.D),'color',D_colour(n,:),'linewidth',2)
    hold all
    plot(diag(Eigy.DB),'color',B_colour(n,:),'linewidth',2)
    %     plot(Data.Dn*ones(2,1),[0,max([diag(Eigy.D);diag(Eigy.DB)])],'k--','linewidth',2);
    %     plot(d*ones(2,1),[0,max([diag(Eigy.D);diag(Eigy.DB)])],'-','color',[.5,.5,.5],'linewidth',2);
    axis square; legend('Eigs_A','Eigs_B')
end
linkaxes(ax,'x'); xlim([0,50]); ylim([0,10])

FormatFig_For_Export(gcf,fontsize,fontname,widths.axis);
print([exportpath 'NR_3_7D_1N'],'-dpdf');

%% Zoom
xlim([0,10]);
print([exportpath 'NR_3_7D_1N_zoom'],'-dpdf');

% Note - 'elbow' looks to be in a consistent place in M_error and D_error,
% with M_error and eigs having an off-by-one step in B vs A case.

%% Plot true d vs Data.Dn for grid of noise and D


clear MB_error DB_error M_error D_error

rng(13);
O = 100;
F = 50;
% d = 4;

NL = linspace(0.01,1,10);
clear All_Data All_eig
for n = 1:10
    for d = 1:10
        clear C D V W B VB DB
        X = [];
        
        for i = 1:d
            X = [X,ones(O,5).*randn(O,1)];
        end
        % Add column of zeros, then noise over the top
        X = [X, zeros(O,F-size(X,2))];
        
        % Vary noise level
        X = X + NL(n)*randn(O,F);
        
        X = X - mean(X);
        
        Data = reject_the_noise(X);
        All_Data{n,d} = Data;
        
        % Get eigs of C for comparison
        C = cov(X);
        [V,D] = eig(C);
        % Flip around
        D = D(linspace(F,1,F),linspace(F,1,F));
        V = V(:,linspace(F,1,F));
        W = X*V;
        
        % Now get eigs of B
        B = Data.A - Data.ExpA;
        [VB,DB] = eig(B);
        
        % Flip around
        DB = DB(linspace(F,1,F),linspace(F,1,F));
        VB = VB(:,linspace(F,1,F));
        
        % Loadings of VB on to the data
        WB = X*VB;
        
        CsB = zeros(F);
        XsB = zeros(F);
        Cs = zeros(F);
        Xs = zeros(F);
        
        for i = 1:F
            CsB = CsB + DB(i,i) * VB(:,i)*VB(:,i)';
            XsB = WB(:,1:i)*VB(:,1:i)';
            
            % Compare to non-spectral rejected version
            Cs = Cs + D(i,i) * V(:,i)*V(:,i)';
            Xs = W(:,1:i)*V(:,1:i)';
            % Error compute
            MB_error(i,n,d) = norm(B-CsB,'fro');
            DB_error(i,n,d) = norm(X-XsB,'fro');
            
            M_error(i,n,d) = norm(C-Cs,'fro');
            D_error(i,n,d) = norm(X-Xs,'fro');
        end
        
        Eigy.D = D;
        Eigy.V = V;
        Eigy.DB = DB;
        Eigy.VB = VB;
        Eigy.W = W;
        Eigy.WB = WB;
        
        All_eig{n,d} = Eigy;
        
    end
end

save('SR_10D_10N.mat','MB_error','DB_error','M_error','D_error','All_Data','All_eig')

%% NOW plot d vs Data.Dn for all 100 datapoints
clear Rejection TrueD

cmap = brewermap(12,'*Purples');
figure('Units', 'centimeters', 'PaperPositionMode', 'auto','Position',[10 15 [10,10]]);

for i  = 1:10
    for j = 1:10
        Data = All_Data{i,j};
        RejectionD(i,j) = Data.Dn;
        TrueD(i,j) = j;
    end
    plot(TrueD(i,:)+0.05*randn(10,1),RejectionD(i,:)+0.05*randn(10,1),'.','markersize',5,'color',cmap(i,:))
    hold all
    
end


plot([0,11],[0,11],'k')
xlim([0,11]); ylim([0,11])
axis square
xlabel('True D')
hold all
ylabel('Rejection D')

FormatFig_For_Export(gcf,fontsize,fontname,widths.axis);
print([exportpath 'RejectDvsTrueD1010b'],'-dpdf');

%% Plot 10 M_errors (all Ds, one N) with markers for True D and Data.Dn
cmap = brewermap(10,'Set3');
figure('Units', 'centimeters', 'PaperPositionMode', 'auto','Position',[10 15 [10,10]]);

clf
% for i = 1:10
    i = 5
    for j = 1:10
        plot(MB_error(:,i,j),'color',cmap(j,:),'linewidth',2);
        hold all
        plot(TrueD(i,j),MB_error(TrueD(i,j),i,j),'o','markersize',10,'color',cmap(j,:))
        plot(RejectionD(i,j),MB_error(RejectionD(i,j),i,j),'.','markersize',20,'color',cmap(j,:))

        %         plot(TrueD(i,j),M_error(TrueD(i,j),i,j),'k.')
    end
% end

axis square
ylabel('MB_{error}')
xlabel('D')
FormatFig_For_Export(gcf,fontsize,fontname,widths.axis);
print([exportpath 'MB_error_TrueD_RejectionD'],'-dpdf');

%% zoom
xlim([0,15])
print([exportpath 'MB_error_TrueD_RejectionD_zoom'],'-dpdf');

%% Plot 10 M_errors (all Ns, one D) with markers for True D and Data.Dn
cmap = brewermap(12,'*PuBu');
figure('Units', 'centimeters', 'PaperPositionMode', 'auto','Position',[10 15 [10,10]]);

clf
for i = 1:10
    j = 5
%     for j = 1:10
        plot(MB_error(:,i,j),'color',cmap(i,:),'linewidth',2);
        hold all
        plot(TrueD(i,j),MB_error(TrueD(i,j),i,j),'o','markersize',10,'color',cmap(i,:))
        plot(RejectionD(i,j),MB_error(RejectionD(i,j),i,j),'.','markersize',20,'color',cmap(i,:))

        %         plot(TrueD(i,j),M_error(TrueD(i,j),i,j),'k.')
%     end
end

axis square
ylabel('MB_{error}')
xlabel('D')
FormatFig_For_Export(gcf,fontsize,fontname,widths.axis);
print([exportpath 'MB_error_TrueD_RejectionD_Noise'],'-dpdf');

%% zoom
xlim([0,10])
print([exportpath 'MB_error_TrueD_RejectionD_Noise_zoom'],'-dpdf');


%% grid of plots 2-10 to show ALL the data
CMAPS = {'*Blues','*Greens','*Greys','*Oranges','*PuRd','*Purples','*Reds','*YlGnBu','*YlOrBr'};

figure('Units', 'centimeters', 'PaperPositionMode', 'auto','Position',[10 15 [20,20]]);

for j = 2:10
    ax(j-1) = subplot(3,3,j-1);
    cmap = brewermap(12,CMAPS{j-1});
     for i = 1:10
        plot(M_error(:,i,j),'color',cmap(i,:),'linewidth',2);
        hold all
        plot(TrueD(i,j),M_error(TrueD(i,j),i,j),'o','markersize',10,'color',cmap(i,:))
        plot(RejectionD(i,j),M_error(RejectionD(i,j),i,j),'.','markersize',20,'color',cmap(i,:))

        %         plot(TrueD(i,j),M_error(TrueD(i,j),i,j),'k.')
     end
     axis square
    ylabel('M_{error}')
    xlabel('D')
    title(['D = ',num2str(j)])
end


FormatFig_For_Export(gcf,fontsize,fontname,widths.axis);
print([exportpath 'M_error_TrueD_RejectionD_Grid'],'-dpdf');

%% Zoom
linkaxes(ax,'x'); xlim([1,11])
print([exportpath 'M_error_TrueD_RejectionD_Grid_zoom'],'-dpdf');

%% Eig version

figure('Units', 'centimeters', 'PaperPositionMode', 'auto','Position',[10 15 [20,20]]);

for j = 2:10
    ax(j-1) = subplot(3,3,j-1);
    cmap = brewermap(12,CMAPS{j-1});
    for i = 1:10
        Eigy = All_eig{i,j};
        e = diag(Eigy.DB);
        plot(e,'color',cmap(i,:),'linewidth',2);
        hold all
        plot(TrueD(i,j),e(TrueD(i,j)),'o','markersize',10,'color',cmap(i,:));
        plot(RejectionD(i,j),e(RejectionD(i,j)),'.','markersize',20,'color',cmap(i,:));
        
        %         plot(TrueD(i,j),M_error(TrueD(i,j),i,j),'k.')
    end
    axis square
    ylabel('EigB')
    xlabel('D')
    title(['D = ',num2str(j)])
end


FormatFig_For_Export(gcf,fontsize,fontname,widths.axis);
print([exportpath 'EigsB_TrueD_RejectionD_Grid'],'-dpdf');

%% Zoom
linkaxes(ax,'x'); xlim([1,11])
print([exportpath 'EigsB_TrueD_RejectionD_Grid_zoom'],'-dpdf');

%% CALCIUM IMAGING DATA
%% M_error, MB_error and cumsum Eig for example calcium data
clear all
data_path = '~/work/Peron_crcns/example_data/';
local_save_path = '~/work/Peron_crcns/shuffle_tests/';

data_ID = {'ca';'ev';'LZ_e';'ML_e';'S2P_e';'Y_e';'LZ_t';'S2P_t10';'ML_t';'ML_p';...
    'LZ_k';'LZ_t2';'S2P_k10';'S2P_k6';'ML_t_hand';'ML_e2';'S2P_t6'}; 
methods = {'ca';'ev';'LZ';'ML';'S2P';'Y';'LZ_t';'S2P_t10';'ML_t';'ML_p';'LZ_k';...
    'LZ_t2';'S2P_k10';'S2P_k6';'ML_t_hand';'ML_e2';'S2P_t6'}; 
meth_names = {'Calcium','Peron','LZero_{kernel_{old}}','MLSpike_{kernel_{hand}}',...
    'Suite2P_{kernel_{hand}}','Yaksi','LZero_{events_{hand}}','Suite2P_{events_{PCC}}',...
    'MLSpike_{events_{ER}}','MLSpike_{pspike}','LZero_{kernel_{ER}}','LZero_{events_{ER}}',...
    'Suite2P_{kernel_{PCC}}','Suite2P_{kernel_{ER}}','MLSpike_{events_{hand}}','MLSpike_{kernel_{ER}}','Suite2P_{events_{ER}}'};

ERmeths = [1,2,6,17,9,12,14,16,11];
meths = ERmeths; nmeths = numel(meths);

clear egs explained MB_error DB_error M_error D_error

F = 1552;

for j = 1 :nmeths 
    clear C D V W B VB DB
    meth = methods{meths(j)}
    % Load data
    load([data_path,data_ID{meths(j)},'.mat'])
%     load(['~/work/Peron_crcns/example_data/',data_ID{j},'.mat'])
    X = eval(data_ID{meths(j)});
    X(find(isnan(X))) = 0;

    X = X - mean(X);
    display('X is ready')
        Data = reject_the_noise(X);
        display('noise is rejected')
        All_Data{n,d} = Data;
        
        % Get eigs of C for comparison
        C = cov(X);
        [V,D] = eig(C);
        
        % Storing data in same format as deconv paper
        egs(j,:) = sort(diag(D),1,'descend');
        explained(j,:) = 100*cumsum(egs(j,:))/sum(egs(j,:));
        
        % Flip around
        D = D(linspace(F,1,F),linspace(F,1,F));
        V = V(:,linspace(F,1,F));
        W = X*V;
        
        % Now get eigs of B
        B = Data.A - Data.ExpA;
        [VB,DB] = eig(B);
        
        % Flip around
        DB = DB(linspace(F,1,F),linspace(F,1,F));
        VB = VB(:,linspace(F,1,F));
        
        % Loadings of VB on to the data
        WB = X*VB;
        
        CsB = zeros(F);
        XsB = zeros(F);
        Cs = zeros(F);
        Xs = zeros(F);
        
        display('Eig decomp done')
        
        for i = 1:F
            CsB = CsB + DB(i,i) * VB(:,i)*VB(:,i)';
            XsB = WB(:,1:i)*VB(:,1:i)';
            
            % Compare to non-spectral rejected version
            Cs = Cs + D(i,i) * V(:,i)*V(:,i)';
            Xs = W(:,1:i)*V(:,1:i)';
            % Error compute
            MB_error(i,j) = norm(B-CsB,'fro');
            DB_error(i,j) = norm(X-XsB,'fro');
            
            M_error(i,j) = norm(C-Cs,'fro');
            D_error(i,j) = norm(X-Xs,'fro');
        end
        
        display('Reconstruction error calculated')
        
        Eigy.D = D;
        Eigy.V = V;
        Eigy.DB = DB;
        Eigy.VB = VB;
        Eigy.W = W;
        Eigy.WB = WB;
        
        All_eig{n,d} = Eigy;
        
end

save('Peron_example.mat','MB_error','DB_error','M_error','D_error','All_Data','All_eig','egs','explained')





%% Plot
figure(20);
clf
for i = 1:nmeths
    plot(explained(i,:),'color',cmap(i,:),'linewidth',2);
    hold all
end
ylabel('Variance explained (%)')
xlabel('N eigenvectors')


% Add line for 80% variance
plot([0,ncells],[80,80],'k--')

% For each method, work out nDims for >80 explained variance
for i = 1:nmeths
    th(i) = find(explained(i,:) >= 80,1,'first');
    plot(th(i)*ones(1,2),[0,80],'color',cmap(i,:))
end
axis square
legend(meth_names(meths),'location','best')
print('deconv_paper_figs/2019_v2/dimensionality','-dsvg')

