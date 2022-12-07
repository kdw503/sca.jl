
using Pkg
import Base:pathof

if Sys.iswindows()
    cd("C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca")
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    cd("/storage1/fs1/holy/Active/daewoo/work/julia/sca")
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end

Pkg.activate(".")

#using MultivariateStats # for ICA
using Images, Convex, SCS, LinearAlgebra, Printf, Colors
using FakeCells, AxisArrays, ImageCore, MappedArrays
using ImageAxes # avoid using ImageCore.nimages for AxisArray type array
using MAT, MLJ
using SymmetricComponentAnalysis
SCA = SymmetricComponentAnalysis

Images.save("dummy.png",rand(2,2)) # If we didn't call this line before `using PyPlot`,
                            # ImageMagick is used when saving image. That doesn't
                            # properly close the file after saving.
using PyPlot
scapath = joinpath(dirname(pathof(SymmetricComponentAnalysis)),"..")
include(joinpath(scapath,"test","testdata.jl"))
include(joinpath(scapath,"test","testutils.jl"))

plt.ioff()

# ARGS = [":ksvd","10000"]
dr_method = eval(Meta.parse(ARGS[1])); num_trials = eval(Meta.parse(ARGS[2]))
@show dr_method, num_trials

frn = "IMAGES.mat"
dd = matread(frn)
img = dd["IMAGES"]
szh,szv,numimgs = size(img)
batch_size = 100; ncomps = 64; subimgsz = 8
L = subimgsz^2
BUFF = 4 # border pixels buffer
W = zeros(L,ncomps)
X = zeros(L,batch_size*num_trials)
# for i=1:numimgs
for t=1:num_trials
    i=Int(ceil(numimgs*rand()))
    this_image=img[:,:,i]
    for j=1:batch_size
        r=Int(BUFF+ceil((szv-subimgsz-2*BUFF)*rand()));
        c=Int(BUFF+ceil((szh-subimgsz-2*BUFF)*rand()));
        X[:,(i-1)*batch_size+j]=reshape(this_image[r:r+subimgsz-1,c:c+subimgsz-1],L,1);
    end
end
if dr_method == :ksvd
    W, H = ksvd(
        X,
        ncomps,  # the number of atoms in D
        max_iter = 200,  # max iterations of K-SVD
        max_iter_mp = 40,  # max iterations of matching pursuit called in K-SVD
        sparsity_allowance = 0.96  # stop iteration when more than 96% of elements in X become zeros
    )
else
    error("Not supported Dimensionality Reduction method")
end
normalizeWH!(W,H)
imgsz = (subimgsz,subimgsz)
clamp_level=0.5; W_max = maximum(abs,W)*clamp_level; W_clamped = clamp.(W,0.,W_max)
signedcolors = (colorant"green1", colorant"white", colorant"magenta")
imsaveW("onoff_natural_$(dr_method)_nc$(ncomps)_nt$(num_trials).png", W, imgsz, gridcols=8, colors=signedcolors, borderval=W_max, borderwidth=1)

#========= MLJ ==============================================#
# load dataset : MLJBase/src/data/dataset.jl
iris = load_iris()

# show the data in table. The dataset provided are compatible with Table.jl (ex. Tables.istable(iris)==true)
selectrows(iris, 1:3)  |> pretty # table view of some data
schema(iris) # fields of the dataset?

# Also, compatible with DataFrames
import DataFrames
iris = DataFrames.DataFrame(iris);

# split the data horizontally
y, X = unpack(iris, ==(:target), x -> x != :target; rng=123); # fn = ==(:target); fn(:target)==true; rng->Random Number Generator seed?
y, X = unpack(iris, ==(:target), !=(:target); rng=123); # unpack :target column then unpack rest of columns which doen't include :target 
y, X = unpack(iris, ==(:target), x -> true; rng=123); # unpack :target column then unpack all the rest of columns
first(X, 3) |> pretty
# small = (x=collect(1:5), y = collect("abcde"))
# x, y = unpack(small, ==(:x), ==(:y); rng=StableRNG(123))

# To list all models available in MLJ's model registry do models(). Listing the models compatible with the present data:
models() # all models
models(matching(X,y)) # all models matched with (X,y)
localmodels() # models for which code is already loaded with
localmodels(matching(X,y))
info("PCA") # display the information of "PCA"
filter(model) = model.is_supervised &&
                       model.input_scitype >: MLJ.Table(Continuous) &&
                       model.target_scitype >: AbstractVector{<:Multiclass{3}} &&
                       model.prediction_type == :deterministic
models(filter)
task(model) = matching(model, X, y) && model.prediction_type == :probabilistic
models(task)

#========= Data ==================================#
# <Data Type>
# Finite{N} : OrderedFactor{N}
#             Multiclass{N}
# Infinite : Continuous
#            Count
# Textual
# Missing
scitype(4.6) # Continuous
scitype(42) # Count
scitype([1,2]) # AbstractVector{Count}
scitype(coerce([1,2],Continuous)) # AbstractVector{Continuous}
scitype(coerce([1,2],Multiclass)) # AbstractVector{Multicalss{2}}
scitype(coerce([1,2],OrderedFactor)) # AbstractVector{OrderedFactor{2}}
scitype("Yes") # Textual
scitype(["Yes","No"]) # AbstractVector{Textual}
scitype(coerce(["Yes","No"],Multiclass)) # AbstractVector{Multicalss{2}}
scitype(coerce(["Yes","No"],OrderedFactor)) # AbstractVector{OrderedFactor{2}}
scitype(coerce(["Yes","No","Yes","No"],Multiclass)) # AbstractVector{Multicalss{2}}
scitype(missing) # Missing
scitype(coerce(["Yes",missing],OrderedFactor)) # AbstractVector{Union{Missing, OrderedFactor{1}}}
using Tables
X = rand(2,3)
scitype(X) # AbstractMatrix{Continuous}
Tables.istable(X) # false
X_tbl = MLJ.table(X)
Tables.istable(X_tbl) # true
scitype(X_tbl) # Table{AbstractVector{Continuous}} Generally, two-dimensional data in MLJ is expected to be tabular
X_dicname = autotype(X_tbl; only_changes=true, rules=(:few_to_finite,)) # return Dictionary of {column nae, type}

# Bridging the gap between data type and model requirements
X = (height   = [185, 153, 163, 114, 180],
     time     = [2.3, 4.5, 4.2, 1.8, 7.1],
     mark     = ["D", "A", "C", "B", "A"],
     admitted = ["yes", "no", missing, "yes"]);
y = [12.4, 12.5, 12.0, 31.9, 43.0]
models(matching(X, y)) # return only model satisfying scitype(X) <: input_scitype(model) 
# Scientific type coercion
schema(X)
`
┌──────────┬────────────────────────┬─────────────────────────┐
│ _.names  │ _.types                │ _.scitypes              │
├──────────┼────────────────────────┼─────────────────────────┤
│ height   │ Int64                  │ Count                   │
│ time     │ Float64                │ Continuous              │
│ mark     │ String                 │ Textual                 │
│ admitted │ Union{Missing, String} │ Union{Missing, Textual} │
└──────────┴────────────────────────┴─────────────────────────┘
`
X_coerced = coerce(X, :height=>Continuous, :mark=>OrderedFactor, :admitted=>Multiclass);
schema(X_coerced)
`
┌──────────┬──────────────────────────────────────────────────┬───────────────────────────────┐
│ _.names  │ _.types                                          │ _.scitypes                    │
├──────────┼──────────────────────────────────────────────────┼───────────────────────────────┤
│ height   │ Float64                                          │ Continuous                    │
│ time     │ Float64                                          │ Continuous                    │
│ mark     │ CategoricalValue{String, UInt32}                 │ OrderedFactor{4}              │
│ admitted │ Union{Missing, CategoricalValue{String, UInt32}} │ Union{Missing, Multiclass{2}} │
└──────────┴──────────────────────────────────────────────────┴───────────────────────────────┘
`
# Data transformations
imputer = FillImputer() # fill missing values
mach = machine(imputer, X_coerced) |> fit!
X_imputed = transform(mach, X_coerced); # missing is replaced by "yes"
schema(X_imputed)
`
┌──────────┬──────────────────────────────────┬──────────────────┐
│ _.names  │ _.types                          │ _.scitypes       │
├──────────┼──────────────────────────────────┼──────────────────┤
│ height   │ Float64                          │ Continuous       │
│ time     │ Float64                          │ Continuous       │
│ mark     │ CategoricalValue{String, UInt32} │ OrderedFactor{4} │
│ admitted │ CategoricalValue{String, UInt32} │ Multiclass{2}    │
└──────────┴──────────────────────────────────┴──────────────────┘
`
encoder = ContinuousEncoder()
mach = machine(encoder, X_imputed) |> fit!
X_encoded = transform(mach, X_imputed)
schema(X_encoded)
`
┌───────────────┬─────────┬────────────┐
│ _.names       │ _.types │ _.scitypes │
├───────────────┼─────────┼────────────┤
│ height        │ Float64 │ Continuous │
│ time          │ Float64 │ Continuous │
│ mark          │ Float64 │ Continuous │
│ admitted__no  │ Float64 │ Continuous │
│ admitted__yes │ Float64 │ Continuous │
└───────────────┴─────────┴────────────┘
`
#=====  Load model and evaluate it with data ==============#
Tree = @load DecisionTreeClassifier pkg=DecisionTree # import the DecisionTreeClassifier model type, and bind to Tree
tree = Tree()
MLJ.evaluate(tree, X, y, resampling=CV(shuffle=true), measures=[log_loss, accuracy], verbosity=0) # this has error 
MLJ.evaluate(tree, X, y, resampling=CV(shuffle=true), measures=[log_loss], verbosity=0)
MLJ.evaluate(tree, X, y, resampling=CV(shuffle=true), verbosity=0)
# Create machine(data + model + training outcomes)
# machine(model::Unsupervised, X) # unsupervised model
# machine(model::Supervised, X, y) # supervised model
supertype(typeof(tree)) # Probabilistic
supertype(supertype(typeof(tree))) # Supervised
mach = machine(tree, X) # This issues an error
mach = machine(tree, X, y)

#========= Workflow ==================================#
import DataFrames
# 1. Data ingestion
iris = load_iris() # load dataset : MLJBase/src/data/dataset.jl
# Then, make compatible with DataFrames (make Table object)
iris = DataFrames.DataFrame(iris);
# Simply, you can combine these two
iris = load_iris() |> DataFrames.DataFrame
# show the data in table. The dataset provided are compatible with Table.jl (ex. Tables.istable(iris)==true)
selectrows(iris, 1:3)  |> pretty # table view of some data
schema(iris) # fields of the dataset?
# split the data horizontally
y, X = unpack(iris, ==(:target), x -> x != :target; rng=123); # fn = ==(:target); fn(:target)==true; rng->Random Number Generator seed?
y, X = unpack(iris, ==(:target), !=(:target); rng=123); # unpack :target column then unpack rest of columns which doen't include :target 
y, X = unpack(iris, ==(:target), x -> true; rng=123); # unpack :target column then unpack all the rest of columns
first(X, 3) |> pretty
# small = (x=collect(1:5), y = collect("abcde"))
# x, y = unpack(small, ==(:x), ==(:y); rng=StableRNG(123))

# Or, just simply, load a built-in data set already split into X and y
X, y = @load_iris |> DataFrames.DataFrame
X = DataFrames.DataFrame(X);

# 2. Model Search
# To list all models available in MLJ's model registry do models(). Listing the models compatible with the present data:
models() # all models
ms = models(matching(X,y)) # all models matched with (X,y)
ms[6]
models("Tree")
localmodels() # models for which code is already loaded with
localmodels(matching(X,y))
info("PCA") # display the information of "PCA"
info("RidgeRegressor", pkg="MultivariateStats") # a model type in multiple packages
# using filter
filter(model) = model.is_supervised &&
                       model.input_scitype >: MLJ.Table(Continuous) &&
                       model.target_scitype >: AbstractVector{<:Multiclass{3}} &&
                       model.prediction_type == :deterministic
models(filter)
task(model) = matching(model, X, y) && model.prediction_type == :probabilistic
models(task)
# more refined search
models() do model
    matching(model, X, y) &&
    model.prediction_type == :deterministic &&
    model.is_pure_julia
end

# 3. Load Model
Tree = @load DecisionTreeClassifier pkg=DecisionTree # import the DecisionTreeClassifier model type, and bind to Tree
tree = Tree(min_samples_split=5, max_depth=4)

# 4. Training data
# Resampling (for holdout set)
train, test = partition(eachindex(y), 0.7); # 70:30 split
# Create machine(data + model + training outcomes)
# machine(model::Unsupervised, X) # unsupervised model
# machine(model::Supervised, X, y) # supervised model
supertype(typeof(tree)) # Probabilistic
supertype(supertype(typeof(tree))) # Supervised
mach = machine(tree, X) # This issues an error because it tried to construct unsupervised machine with supervised model
mach = machine(tree, X, y)
# Fit
# Hyper-parameters : rng=Random.GLOBAL_RNG (random number generator), display_depth, model specific options ...
fit!(mach, rows=train)

# 5. Prediction and Evaluation
# Predict
yhat = MLJ.predict(mach, X[test,:]); # probabilities for each mode("setosa", "versicolor", "virginica" in this example)
# Evaluate
measures() # list all the measurement types
log_loss(yhat, y[test]) |> mean
typeof(yhat) # UnivariateFiniteVector{Multiclass{3}, String, UInt32, Float64} -> vector of Distribution objects
pdf.(yhat[3:5], "virginica")
pdf.(yhat, y[test])[3:5]
mode.(yhat[3:5]) # just choose one mode which is most probable
predict_mode(mach, X[test[3:5],:]) # predic and choose mode
L = levels(y) # all the levels = ["setosa", "versicolor", "virginica"]
pdf(yhat[3:5], L) # all probabilities for the levels
mean(LogLoss(tol=1e-4)(yhat, y[test]))
# Evaluate with evalute! fn : ! is needed because mach will be changed
MLJ.evaluate!(mach, resampling=Holdout(fraction_train=0.7), # Holdout
                    measures=[log_loss],                    # Resampling + Fit + Predict + Evaluate
                    verbosity=0)
MLJ.evaluate!(mach, resampling=CV((nfolds=5, shuffle=true, rng=1234)), # Cross-Validation
                    measures=[log_loss],                    # Resampling + Fit + Predict + Evaluate
                    verbosity=0)
MLJ.evaluate(tree, X, y, resampling=CV(shuffle=true), # Create Machine + Resampling + Fit + Predict + Evaluate
                         measures=[log_loss],
                         verbosity=0)
# Changing a hyperparameter and re-evaluating
tree.max_depth = 3
evaluate!(mach, resampling=Holdout(fraction_train=0.7),
                 measures=[log_loss, accuracy],
                 verbosity=0)
# Fitted parameters
fitted_params(mach).tree
`
Decision Tree
Leaves: 4
Depth:  3
`
fitted_params(mach).encoding
`
Dict{CategoricalArrays.CategoricalValue{String, UInt32}, UInt32} with 3 entries:
  "virginica"  => 0x00000003
  "setosa"     => 0x00000001
  "versicolor" => 0x00000002
`
# Report
report(mach).print_tree(3)
`
Feature 3 < 2.45 ?
├─ 1 : 50/50
└─ Feature 4 < 1.75 ?
    ├─ 2 : 49/49
    └─ Feature 3 < 4.95 ?
        ├─ 2 : 1/1
        └─ 3 : 5/5
`

#========= transform and inverse_transform ==================================#
# Instead of predict, transform method is used for the unsupervised models
v = Float64[1,2,3,4] # data
stand = Standardizer() # model
mach2 = machine(stand, v) # machine
fit!(mach2)
w = transform(mach2, v)
inverse_transform(mach2, w)

#========= user specified resampling =================#
f1, f2, f3 = 1:13, 14:26, 27:36
pairs = [(f1, vcat(f2, f3)), (f2, vcat(f3, f1)), (f3, vcat(f1, f2))];
evaluate!(mach,
          resampling=pairs,
          measure=[LogLoss(), Accuracy()])
#======== Tuning Models ===========================#
"""
MLJ provides several built-in and third-party options for optimizing a model's hyper-parameters.
The quick-reference table below omits some advanced keyword options.
Grid, RandomSearch, LatinHypercube, ParticleSwarm, Explicit
"""
X = MLJ.table(rand(100, 10));
y = 2X.x1 - X.x2 + 0.05*rand(100);
Tree = @load DecisionTreeRegressor pkg=DecisionTree verbosity=0;
tree = Tree()
`
DecisionTreeRegressor(
  max_depth = -1, 
  min_samples_leaf = 5, 
  min_samples_split = 2, 
  min_purity_increase = 0.0, 
  n_subfeatures = 0, 
  post_prune = false, 
  merge_purity_threshold = 1.0, 
  feature_importance = :impurity, 
  rng = Random._GLOBAL_RNG())
`
# Want to tune min_purity_increase hyperparameter above model
r = range(tree, :min_purity_increase, lower=0.001, upper=1.0, scale=:log);
self_tuning_tree = TunedModel(model=tree,
							  resampling=CV(nfolds=3),
							  tuning=Grid(resolution=10), # tuning method
							  range=r,
							  measure=rms);
mach = machine(self_tuning_tree, X, y)
fit!(mach)
F = fitted_params(mach)
F.best_model
`
DecisionTreeRegressor(
  max_depth = -1, 
  min_samples_leaf = 5, 
  min_samples_split = 2, 
  min_purity_increase = 0.004641588833612781, 
  n_subfeatures = 0, 
  post_prune = false, 
  merge_purity_threshold = 1.0, 
  feature_importance = :impurity, 
  rng = Random._GLOBAL_RNG())
`
# Inspecting details of tuning procedure:
r = report(mach);
keys(r)
`
(:best_model, :best_history_entry, :history, :best_report, :plotting)
`
r.history[[1,end]]

#========= Composing Models =====================#
"""
Three common ways of combining multiple models together have out-of-the-box implementations in MLJ:

* Linear Pipelines (Pipeline)- for unbranching chains that take the output of one model (e.g., dimension
reduction, such as PCA) and make it the input of the next model in the chain (e.g., a classification model,
such as EvoTreeClassifier). To include transformations of the target variable in a supervised pipeline model,
see Target Transformations.
* Homogeneous Ensembles (EnsembleModel) - for blending the predictions of multiple supervised models all of
the same type, but which receive different views of the training data to reduce overall variance. The
technique implemented here is known as observation bagging.
* Model Stacking - (Stack) for combining the predictions of a smaller number of models of possibly different
types, with the help of an adjudicating model.

Additionally, more complicated model compositions are possible using:

Learning Networks - "blueprints" for combining models in flexible ways; these are simple transformations of
your existing workflows which can be "exported" to define new, stand-alone model types.
"""
# Linear pipeline
X, y = @load_reduced_ames
KNN = @load KNNRegressor
knn_with_target = TransformedTargetModel(model=KNN(K=3), target=Standardizer()) # 
pipe = (X -> coerce(X, :age=>Continuous)) |> OneHotEncoder() |> knn_with_target
`
DeterministicPipeline(
  f = Main.var"ex-workflows".var"#15#16"(), 
  one_hot_encoder = OneHotEncoder(
        features = Symbol[], 
        drop_last = false, 
        ordered_factor = true, 
        ignore = false), 
  transformed_target_model_deterministic = TransformedTargetModelDeterministic(
        model = KNNRegressor(K = 3, …), 
        transformer = Standardizer(features = Symbol[], …), 
        inverse = nothing, 
        cache = true), 
  cache = true)
`
pipe.one_hot_encoder.drop_last = true # Hyperparameter setting
evaluate(pipe, X, y, resampling=Holdout(), measure=RootMeanSquaredError(), verbosity=2) # evaluate
mach = machine(pipe, X, y) |> fit!
F = fitted_params(mach)
F.transformed_target_model_deterministic.model

# Constructing a linear (unbranching) pipeline with a static (unlearned) target transformation/inverse transformation:
Tree = @load DecisionTreeRegressor pkg=DecisionTree verbosity=0
tree_with_target = TransformedTargetModel(model=Tree(),
                                          target=y -> log.(y),
                                          inverse = z -> exp.(z))
pipe2 = (X -> coerce(X, :age=>Continuous)) |> OneHotEncoder() |> tree_with_target;

# Creating a homogeneous ensemble of models
X, y = @load_iris
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()
forest = EnsembleModel(model=tree, bagging_fraction=0.8, n=300)
mach = machine(forest, X, y)
evaluate!(mach, measure=LogLoss())

#=========== Performance curves ====================#
"""
Generate a plot of performance, as a function of some hyperparameter (building on the preceding example)
"""
# Single performance curve:
r = range(forest, :n, lower=1, upper=1000, scale=:log10)
curve = learning_curve(mach,
                       range=r,
                       resampling=Holdout(),
                       resolution=50,
                       measure=LogLoss(),
                       verbosity=0)
using Plots
plot(curve.parameter_values, curve.measurements, xlab=curve.parameter_name, xscale=curve.parameter_scale) 

# Multiple curves:
curve = learning_curve(mach,
                       range=r,
                       resampling=Holdout(),
                       measure=LogLoss(),
                       resolution=50,
                       rng_name=:rng, #
                       rngs=4, #
                       verbosity=0)
plot(curve.parameter_values, curve.measurements, xlab=curve.parameter_name, xscale=curve.parameter_scale)

#========= Examples of different models ==================================#
# KNN
using NearestNeighborModels
X, y = @load_boston
KNN = @load KNNRegressor
knn = KNN()
MLJ.evaluate(knn, X, y,
         resampling=CV(nfolds=5),
         measure=[MLJ.RootMeanSquaredError(), MLJ.MeanAbsoluteError()])
# PCA
using MultivariateStats, MLJMultivariateStatsInterface
X, y = @load_iris
train, test = partition(eachindex(y), 0.97, shuffle=true, rng=123)
PCA = @load PCA
pca = PCA(maxoutdim=2)
mach = machine(pca, X)
fit!(mach, rows=train)
MLJ.transform(mach, rows=test) # transfrom test rows
Xnew = (sepal_length=rand(3), sepal_width=rand(3), # New data
        petal_length=rand(3), petal_width=rand(3));
MLJ.transform(mach, Xnew)
