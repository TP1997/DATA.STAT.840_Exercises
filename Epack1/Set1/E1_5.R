
multivar_gaussian = function(x, mu, sigma){
  n = length(x)
  p1 = 1 / (sqrt((2*pi)^n * det(sigma)))
  p2 = t(x - mu) %*% solve(sigma) %*% (x - mu)
  p2 = exp(-0.5 * p2)
  
  return(p1 * p2)
}

xs = list(c(2,2,2), c(1,4,3), c(1,1,5))
mu = c(1,3,5)
sigma = matrix(c(4,2,1,2,5,2,1,2,3), nrow = 3, byrow = T)

res1 = unlist(lapply(xs, multivar_gaussian, mu=mu, sigma=sigma))

res2 = lapply(xs, dmvnorm, mean=mu, sigma=sigma)
