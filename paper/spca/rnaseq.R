library(scRNAseq)
library(magrittr)
library(SingleCellExperiment)
library(dplyr)
library(epca)
library(Matrix) # as.matrix
library(ggplot2)

dataset <- "Baron" # Baron, Muraro

dataset_functions <- list(
  "Muraro" = MuraroPancreasData,
  "Baron" = BaronPancreasData
)

# Check if dataset exists and call the correct function
if (dataset %in% names(dataset_functions)) {
  cat(dataset, "dataset\n")
  dat <- dataset_functions[[dataset]]()
} else {
  stop("Unknown dataset: ", dataset)
}
dim(dat)
names(assays(dat))

gene.select <- !!apply(counts(dat), 1, sd) # select genes with high variance
label.select <- colData(dat) %>% # select labels with more than 100 cells
                data.frame() %>%
                dplyr::count(label) %>%
                filter(n > 100)
dat1 <- dat[gene.select, colData(dat)$label %in% label.select$label]
label <- setNames(factor(data.frame(colData(dat1))$label), colnames(dat1)) # cell type label
count <- counts(dat1)
genename <- as.matrix(rownames(dat1))   # gene names

noc <- 9 # Baron, 7(Muraro)
start <- Sys.time()
scar <- sca(t(count), k = noc, gamma = 1e5,
               center = F, scale = F,
               epsilon = 1e-3)
runtime <- Sys.time() - start
n.gene <- apply(!!scar$loadings, 2, sum) # sum of non zero loadings number for each gene
ngene_sma <- n.gene
Wsma <- as.matrix(scar$scores)
Htsma <- as.matrix(scar$loadings)


# initialization only
start <- Sys.time()
x = scale(x = t(count),
            center = F,
            scale = F)
# s = RSpectra::svds(x, k)
s = irlba::irlba(x, noc, tol = 1e-10)
z = s$u
b = diag(s$d)
y = s$v
score = sqrt(sum(s$d ^ 2))
diff = c(z = Inf, y = Inf)
init_runtime <- Sys.time() - start
