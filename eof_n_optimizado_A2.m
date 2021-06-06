% This function makes a  Empirical Orthogonal Functions (EOF) Analysis,
% such as it is described in:: 
% "A manual for EOF and SVD Analysis of Climatic Data" 
% de Bjornsson and Venegas (pag. 10)
%
% Input:        A 3D matrix of data, data(lat, long, t), where "lat" represents the latitude,
%               "long" represents the lontituds and "t" the time (matrix with this shape:
%               tp_9207_0206, sst_9207_0206, etc). We reshape the data in a 2D matrix as 
%               described in Preisendorfer's book, page 25. That is, a 2D matrix where each
%               column is the time series of a map point (map, t).
%
% Output:       "vp": Vector with the eigenvalues
%               "var_porc": Vector with the percentage of the variance explained by each EOF
%               "eof_p_n": Matrix whose columns are the EOF patterns (maps), which are given 
%                 in the same order than the eigenvalues in "vp"
%               "exp_coef_n": Matrix whose columns are the expansion coefficients (time series) 
%                associated to each EOF pattern (map)
%
% The function shape is as follows: [vp, var_porc, eof_p, exp_coef] = eof_n_optimizado(data, ind)
%
% Each EOF is normalized by dividing it by its standard deviation. Then, the associated time series, 
% the expansion coefficient, is multiplied by the same value.
%
% The EOF is calculated in an optimal way in the case that there are much more 
% spatial points than temporal samples. The algorithm is based in 
% Preisendorfer "Principal component analysis in metereology and
% oceanography", page 64. 

function [vp, var_porc, eof_p_n, exp_coef_n] = eof_n_optimizado_A(data)
%data(map, t)
% We reshape the data in a 2x2 matrix as described in Preisendorfer's book, page 25. 
% That is, a 2x2 matrix where each column is the time series of a map point.
Matrix=data;

% We remove the mean value of each time series
Z=detrend(Matrix,'constant'); %removes the mean value from each column of the matrix
% Z=Matrix;

% We estimate the matrix dimensions: 
% n = number of temporal samples
% p = number of points with data
[n,p] = size(Z);

% Accordingly to Preisendorfer's book, pages 64-66:
% If n>=p we estimate the EOF in the State Space Setting, i.e., using Z'*Z which is p x p.
% If n<p  we estimate the EOF in the Sample Space Setting, i.e., using Z*Z' which is n x n 
if n>=p
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% State Space Setting %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Covariance matrix
    R=Z'*Z;
    
    % We estimate the eigenvalues and eigenvectors of the covariance matrix.
    [eof_p,L]=eigs(R, eye(size(R)),10);       % The eigenvelues are in the diagonal of L
                            % and the associated eigenvectors are in the columns of eof_p

    vp=diag(L);                    
    
    % We estimate the expansion coefficients (time series) associated to each EOF
    [a,b]=size(L);
    for i=1:a
        exp_coef(:,i)= Z * eof_p(:,i);
    end
    
elseif n<p
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Sample Space Setting %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fprintf('Sample space setting\n');
    % Scatter matrix
    R=Z*Z';
    
    % We estimate the eigenvalues and eigenvectors of the scatter matrix.
    [eof_p_f,L]=eigs(R, eye(size(R)),10);     % The eigenvelues are in the diagonal of M
                            % and the associated eigenvectors are in the columns of eof_p_f
                            
    vp=diag(L);
    
    % We estimate the expansion coefficients (time series) associated to each EOF
    [a,b]=size(L);
    for i=1:a
        exp_coef_b(:,i)= Z' * eof_p_f(:,i);
    end
    
    % We transform the EOF and their associated expansion coefficient (time series) in the Sample 
    % space setting to those in the State space setting
    
    %Expansion coefficients
    for i=1:a
        eof_p(:,i)= exp_coef_b(:,i) / sqrt(vp(i));
    end
    
    %Expansion coefficients
    for i=1:a
        exp_coef(:,i)= sqrt(vp(i)) * eof_p_f(:,i);
    end
    
end



% We estimate the percentage of variance explained by each EOF/eigenvector
var_porc = ( diag(L)/trace(L) )*100;

%We normalized the EOF
[a_1,b_1]=size(eof_p);
[a_11,b_11]=size(exp_coef);
eof_p_n=eof_p;
exp_coef_n=exp_coef;

for i=1:b_1
    
%     eof_p_n(:,i) = eof_p(:,i)/ rms(eof_p(:,i));
%     exp_coef_n(:,i) = exp_coef(:,i)* rms(eof_p(:,i));
    eof_p_n(:,i) = eof_p(:,i)/ std(eof_p(:,i));
    exp_coef_n(:,i) = exp_coef(:,i) * std(exp_coef(:,i));

    
end



