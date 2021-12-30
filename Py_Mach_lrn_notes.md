# Python Machin Learning by xxx and yyyy

notes from the book 

- convention - underscore after a attribute indicates that it is not created at init time (p 39)
- normalization procedcure helps gradient descent learning to covnergae more quickly (p 50)
- standardization helps with gradient descent learning b/c optimizer has to go through fewer steps (p 51)
- SSE may be non-zero even though all examples are classified - i.e. the model may report some amount of error even with 100% classification (p 51)
- stochastic part of SGD == weights are updated incrementally with each training example (p 52)
- error surface is noisier than in gradient descent
- shuffle the training dataset for every epoch
- adaptive learning rate with SGD rather than fixed learning rate
- online learning == model is trained on the fly as new data arrives
- partial_fit - method which does not reinitialize the weights for online learning
- train_test_split automatically shuffles the data
- startification - training and test subsets containing the same proportions of class labels (supervised)
- plot_decision_regions (algorithm by book authors on p60
- np.ravel (flatten a iter of iters) (p 54)
- OvR = One vs Rest - 
- **all allgorithms require informitive and discrimative features to yield useful predictions**

## logistic regression: easy to implement and performs well on linearly separable classes
  - logit function == logarithm of odds
  - log is used to refer to ln (natural log) thorughout the book :question:
  - uppercase pi symobl is like summation but multiplies over the specified range
  - log reduces the potential for numerical underflow - which can happen if the likelihoods are small
  - sklearn.linear_model.logisticRegression
  - solver paramter allows for different optimization algorithms: newton-cg, lbfgs(default in sklearn 0.22), lib-linear (default in sklearn < 0.22), sag, saga
  - np.argmax to get the largest column in each row - use for the labels
  - for predict sklearn expects a 2d matrix so must convert single row for input using reshape(1,-1)
  - goal is to maximize the conditional likelihoods which may increase outlier impact
- bias variance trade-off
  - high bias == underfitting == error (distance of predictions from actuals) - model is too simple to capture the underlying patterns
  - high variance == overfitting == consistency - works well on train but not on test
    - too many parameters?
  - regularization penalizes extreme paramater weights (lowers bias)
    - L1 regularization - (y2-y1) + (x2-x1)
    - L2 regularization == shrinkage == weight decay - distance formula - sqrt((y2-y1)^2+(x2-x1)^2)
    - requires all features to have comparable scale
    - lambda is regularization parameter sometimes specified as its inverse 'C'

## SVM = support vector machine
  - maximize the margin - the distance the decision boundary and data points
  - support vectors are the data points that are closest to the hyperplane boundary (they drive its location)
  - small margins can indicate overfitting
  - quadratic programming - what is this :question:
  - slack variable \gamma lowercase gamma -used for soft-margin classification - handles cases when data is not perfectly linearly separable
    - set C to control misclassification penalty
  - less prone to outliers than logistic regression
  - liblinear/libsvm == optimized library for solving logistic/SVM problems - work well if all data fits in memory - otherwise use SGDClassifer implementations
- kernel SVM = for nonlinear problems
  - kernel trick - protect the data into a higher dimensional space where it will be linearly separable
  - kernel is a  similarity function between a pair of examples
  - uses a mapping function represented by \phi lowercase phi
  - RBF (radial bias function) == Gaussian kernel - a widely used kernel 
  - cut-off paramater \gamma lowercase gamma - for gaussian sphere - higher menas bumpier decision boundary to better capture training but may lead to overfitting

## Decision Trees - 
explainable - help with dim reduction - no need to regularize inputs
- split based on the largest information gain
- prune == set maximum depth of the tree
- binary trees - to reduce the decision search space
- information gain - reduction in entropy from transforming a dataset - compare entropy before/after the change
  - mutual information - statistical dependence between two variables - name for information gain applied to a variable selection
- impurity measures - usually pruning cutoffs are more helpful (gini is usually roughly equivalent to entropy (scaled?))
  - gini impurity - minimize the probability of missclassification - usually falls between entropy and classification error
  - entropy - maximize mutual information ()
  - classification error - useful for pruning only (not growing) - less sensitive 
  - visualize with `sklearn.tree.plot_tree(model_name)`
  - graphviz has prettier tree visualizations but has many dependencies: better layout, colors 
    - outfile=None to bypass disk and save data to a variable instead
  - PydDotPlus - similar to graphviz and can convert \*.dot files to an image
  - graphviz has useful visualizations for decision trees


## Random forest
large number of small trees, based on random samples from input data
- random sample of data (with replacement)
- random subset of features chosen (without replacement) for each tree
- less interpretable than decision trees but less dependent on hyperparamter tuning
- bias variance tradeoff is controlled by the sample size
- d = number of features to consider at  each split = good starting point is sqrt(m) where m is the number of features in the training dataset
- 

## K-neaest neighbors
lazy learning - memorizes the training data rather than learning a discriminant function - very susceptible to overfitting
- parametrics models = estimate parameters from training data, can discard training data e.g. peceptron, logistic regression, SVM
- non-parametric models = number of parameters grows with training data, training data retained e.g. decision tree, kernel SVM
  -  instance based learning - subset of non-parametric - memorize training data (lazy)
-  for each new datapoint assign the new class by finding the k nearest points and the majority class is chosen for the new point
-  immediately adapts as new data is collected
-  right choice for _k_ is crucial for bias variance trade-off
-  

## data preprocessing
- find missing - fill(fillna) or remove(dropna) or Impute
  - sklearn.impute.SimpleImputer - mean imputation - mean of entire column - 
    - transformer class (fit and transform methods): run fit on train X data features (no y targets) - run transform on both train and test features (and prod)
- numpy vs pandas - sklearn has some support for DataFrames
  - sklearn support for np arrays is more mature - use when possible
- categorical data
  - ordinal - inherent order to the labels (e.g. Small Medium Large) - assign numbers based on index or mapping dictionary
    - sklearn.preprocessing.LabelEncoder later can use inverese_transform method to get labels back
  - nominal - no inherent order (e.g. red, blue, green) - require encoding
    - create columns from the values and code with 0/1
    - 1 column is typically removed as it is inherent from other selections and can cause multi-collinearity of the features
      - multi-collinearity causes matrices to be computationally difficult to invert
    - one-hot encoding - new dummy feature for each unique value in the nominal column 
      - sklearn.preprocessing.One-HotEncoder or sklearn.compose.ColumnTranformer to process mutliple columns at once
      - or pandas get_dummies method - drop_first = True to drop one of the categories
- splitting data - sklearn.model_selection.train_test_split()
  - stratify (see above)
  - common ratios train:test - 60/40 or 70/30 for small datasets - for large datasets it is common to have 90:10 or 99:1
  - common to re-train on whole dataset prior to deployment
- **feature scaling**
  - normalization - rescale features to range 0-1 (special case of min-max scaling) - sklearn.preprocesing.MinMaxScaler
  - standardization - alter the values to have unit variance and 0 mean - (subtract mean and divide by stddev) |x| might be larger than 1 - sklearn.preprocessing.StandardScaler
  - Robust scaling - scaled data according the 1st and 3rd quartiles (i.e. IQR) - sklearn.preprocessing.RobustScaler - recommended for small datasets with many outliers
- **Feature selection**
  - impose a penalty for models with large numbers of features
  - logistic reg with L1 regularization inherently causes weights to go to zero for less signficiant features - minimize model cost function and complexity penalty together
    - lbfgs does not support L1-regularized
    - get feature weights from lr.coef_
  - logistic reg with L2 regularization also causes reduction in weights but not as sparse
    - C is the inverse of the regularization parameter \lambda lambda
  - sequential feature selection
    - sequential backward selection - not implemented in sklearn but reasonably easy to code from scratch
      - Greedy algorithms - locally optimal selections at each stage
      - exhaustive search algorithms - check all possible combinations - more computation - more accurate
    - recursive backward elimination - 
    - tree based methods - feature_importances_ attribute after fitting RandomForestClassifier
  - if 2 features are highly correlated one may be ranked highly while the other is not fully captured
  - sklearn SelectFromModel selects features based on a user threshold (e.g. given a RandomForest and a threshold it will return features)

## Dimensionality reduction
- PCA - principal component analysis - unsupervised method - orthogonal vectors of maximum variance for a desired feature count
  - highly sensitive to data scaling
  - eigenvalues and eigenvectors based on covariance matrix of principal comopnents
    - np.linalg.eig (eigh avoids complex results that may come up with eig)
    - eigenvectors are typically scaled to unit length
  - total vs explained variance - bar chart of variance explained for each component and a cumsum total plotted as a line
  - signs for eigenvector matrix may be flipped depending on the LAPACK implementation on current system (does not effect the model) - mutliply by -1 to fix
  - sklearn.decompisition.PCA
    - setting n_components_ = None will return all components in sorted order instead of doing the dimensionality reduction
- LDA - Linear Discriminant analysis - supervised - find feature subspace the optimizes class seperability
  - assumes normally distributed data and that classes have identical covariance matrices
  - class labels taken into account in the form of mean vectors -computed for each class used for within-class and between-class scatter matrices
  - computing scatter matrix is the same as computing covariance matrix  (covariance matrix is normalized version)
  - instead of performing eigen decompoisition on the covariance matrix the eigenvalue problem is solved directly
- KPCA - kernel PCA
  -  use kernel function to map data into higher dimensional space that is linearly seperable - computationally expensive
  - results are projected onto the feature space directly without a transformation matrix
  - need to center the kernel matrix to guarantee that the feature space is centered at zero
  - most comon kernels
    - polynomial kernel
    - readial basis function (RBF) == Gaussian (see above)
  - not implemented in sklearn :question:
    - book solution uses scipy.spatial.distance.pdist and squareform; sciopy.exp; and scipy.linalg.eigh
  - must experiment to find the right value of gamma \gamma
  - creating datasets for testing
    - sklearn.datasets.make_moons
    - sklearn.datasets.make_circles
    - hyperbolic tangent == sigmoid kernel


## other ensemble methods (Ch 7)
- bagging - 
- boosting - 
