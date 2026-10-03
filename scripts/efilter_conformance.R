# Execute only the named author function, preserving its AST unchanged.
# Other funcs.R expressions import optional packages and simulate data; not needed.
args <- commandArgs(trailingOnly=TRUE)
expressions <- parse("external/efilter/funcs.R")
matches <- Filter(function(x) is.call(x) && identical(x[[1]], as.name("<-")) &&
                    identical(x[[2]], as.name("e_filter")), as.list(expressions))
stopifnot(length(matches)==1)
eval(matches[[1]])
inputs <- as.matrix(read.table(args[1], header=FALSE))
result <- e_filter(inputs[,1], inputs[,2], nrow(inputs), .025, "FDR")
write.table(cbind(result$evals,result$decisions),args[2],row.names=FALSE,col.names=FALSE)
