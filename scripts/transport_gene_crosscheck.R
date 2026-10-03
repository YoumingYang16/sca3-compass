# Independent base-R reconstruction of all frozen signed-gene tests, including
# missing entries kept in the BY family. Inputs are public-derived TPM, not scores.
args <- commandArgs(trailingOnly = TRUE)
x <- as.matrix(read.table(args[1], header = FALSE, na.strings = "nan"))
direction <- scan(args[2], quiet = TRUE)
y <- log2(x + 0.5)
labels <- combn(12, 6)
contrast <- sapply(seq_len(ncol(labels)), function(j) {
  a <- labels[, j]
  rowMeans(y[, a, drop = FALSE]) - rowMeans(y[, -a, drop = FALSE])
})
effect <- rowMeans(y[, 7:12, drop = FALSE]) - rowMeans(y[, 1:6, drop = FALSE])
missing <- !is.finite(effect)
p <- rep(1, length(effect))
for (g in which(!missing)) {
  p[g] <- mean(direction[g] * contrast[g, ] >= direction[g] * effect[g] - 1e-12)
}
q <- p.adjust(p, method = "BY")
write.table(cbind(effect, p, q, as.integer(missing)), args[3], row.names = FALSE,
            col.names = FALSE, quote = FALSE, na = "nan")
