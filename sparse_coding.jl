

#========== On/Off filtered natural image test ==============#
using MAT

frn = "IMAGES.mat"
dd = matread(frn)
img = dd["IMAGES"]
frn = "IMAGES_RAW.mat"
dd = matread(frn)
imgr = dd["IMAGESr"]
fwn = "test.mat"
matwrite(matfilename, Dict("IMAGE" => img, "IMAGESr" => imgr))
size(img,3)

# cgf_fitS.m
function S = cgf_fitS(A,X,noise_var, beta, sigma, tol, ...
    disp_ocbsol, disp_patnum, disp_stats)
% cgf_fitS -- fit internal vars S to the data X using fast congugate gradient
%   Usage
%     S = cgf_fitS(A,X,noise_var,beta,sigma,
%                  [tol, disp_ocbsol, disp_patnum, disp_stats])
%   Inputs
%      A             basis functions
%      X             data vectors
%      noise_var     variance of the noise (|x-As|^2)
%      beta          steepness term for prior
%      sigma         scaling term for prior
%      tol           solution tolerance (default 0.001)
%      disp_ocbsol   display info from the fitting process
%      disp_patnum   display the pattern number
%      disp_stats    display summary statistics for the fit
%   Outputs
%      S             the estimated coefficients

maxiter=100;

[L,M] = size(A);
N = size(X,2);

if ~exist('tol','var');		tol = 0.001;			end
if ~exist('disp_ocbsol','var');	disp_ocbsol = 0;		end
if ~exist('disp_patnum','var');	disp_patnum = 1;		end
if ~exist('disp_stats','var');	disp_stats = 0;			end

Sinit=A'*X;
normA2=sum(A.*A)';
for i=1:N
Sinit(:,i)=Sinit(:,i)./normA2;
end

lambda=1/noise_var;

S = zeros(M,N);
tic
[S niters nf ng] = cgf(A,X,Sinit,lambda,beta,sigma,tol,maxiter,...
disp_ocbsol,disp_patnum);
t = toc;

if (disp_stats)
fprintf(' aits=%6.2f af=%6.2f ag=%6.2f  at=%7.4f\n', ...
niters/N, nf/N, ng/N, t/N);
end

# nmf_fitS.jl
function nmf_fitS!(A,X, noise_var, α; tol=0.001, maxiter=100, disp_stats=false)
    (L, M) = size(A);
    N = size(X,2);

    Sinit=A'*X;
    normA2=sum(A.*A)';
    for i=1:N
        Sinit[:,i]=Sinit[:,i]./normA2;
    end
    S = copy(Sinit)

    t = @elapsed NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=maxiter, α=α, l₁ratio=0.5), X, A, S)
    normalizeWH!(A,S)
    
    if (disp_stats)
        @printf("at=%7.4f\n", t/N)
    end
    S
end

# KSVD.jl : K-SVD (derive W and H from X) and Matching Pursuit (derive H from X and W)
using KSVD

function mp_fitS!(A,X, noise_var, α; tol=0.001, maxiter=100, disp_stats=false)
    (L, M) = size(A);
    N = size(X,2);

    t = @elapsed S = matching_pursuit(X,A,max_iter=maxiter)
    normalizeWH!(A,S)
    
    if (disp_stats)
        @printf("at=%7.4f\n", t/N)
    end
    S
end

# sparsenet.m
using MAT

frn = "IMAGES.mat"
dd = matread(frn)
img = dd["IMAGES"]

A = rand(64,64).-0.5;
A = A*Diagonal(vec(1 ./sqrt.(sum(A.*A,dims=1))));

num_trials=1000;
batch_size=100;

num_images=size(img,3);
image_size=512
BUFF=4;

L, M =size(A);
sz=Int(floor(sqrt(L)));

eta = 1.0;
noise_var= 0.01;
beta= 2.2;
sigma=0.316;
tol=.01;

VAR_GOAL=0.1;
S_var=VAR_GOAL*ones(M,1);
var_eta=.001;
alpha=.02;
gain=sqrt.(sum(A.*A,dims=2))';

X=zeros(L,batch_size);

for t=1:num_trials
    @show t
    # choose an image for this batch
    i=Int(ceil(num_images*rand()));
    this_image=img[:,:,i] #reshape(IMAGES(:,i),image_size,image_size)';
    
    # extract subimages at random from this image to make data vector X(64 X batch_size)
    for i=1:batch_size
        r=Int(BUFF+ceil((image_size-sz-2*BUFF)*rand()));
        c=Int(BUFF+ceil((image_size-sz-2*BUFF)*rand()));
        X[:,i]=reshape(this_image[r:r+sz-1,c:c+sz-1],L,1);
    end

    # calculate coefficients for these data via conjugate gradient routine
    # nmf here!, A is basis vectors(factor) and S is coeeficient matrix(encoding vectors)
    # A(64X64), S(64Xbatch_size)
    S=mp_fitS!(A,X,noise_var,0.1; tol=tol);

    # calculate residual error
    E=X-A*S;

    # update bases
    dA=zeros(L,M);
    for i=1:batch_size
        dA = dA + E[:,i]*S[:,i]';
    end
    dA = dA/batch_size;
    A = A + eta*dA;

    # normalize bases to match desired output variance
    for i=1:batch_size
        S_var = (1-var_eta)*S_var + var_eta*S[:,i].*S[:,i];
    end
    gain = gain .* ((S_var/VAR_GOAL).^alpha);
    normA=sqrt.(sum(A.*A,dims=2));
    for i=1:M
        A[:,i]=gain[i]*A[:,i]/normA[i];
    end

end
imgsz = (8,8)
clamp_level=0.5; A_max = maximum(abs,A)*clamp_level; A_clamped = clamp.(A,0.,A_max)
signedcolors = (colorant"green1", colorant"white", colorant"magenta")
imsaveW("onoff_natural_MP_nt$(num_trials).png", A, imgsz, gridcols=8, colors=signedcolors, borderval=A_max, borderwidth=1)
