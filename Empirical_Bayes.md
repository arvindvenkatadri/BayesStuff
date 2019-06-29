Empirical Bayes
================
Arvind Venkatadri
25/06/2019

Following David Robinson’s book “Empirical Bayesian Estimation”.

Empirical Bayes is an approximation to more exact Bayesian methods -
where a beta distribution is fitted on all available observations and is
then used to improve each individually. What’s great about this method
is that as long as you have a lot of examples, you don’t need to bring
in prior expectations.

## The Beta Distribution

In Bayesian estimation, we have a `prior` probability of an event, which
is updated when a new data point arrives. This updated probability
distribution, `posterior` becomes the prior for the next interation
using the next data point.

The beta distribution is a good choice for Bayesian priors:  
\- domain is \[0,1\] - \> The beta distribution is representing a
probability distribution of probabilities.  
\- Bayesian update to posteriors is very easy

\[
p_{beta}(x) = \frac{x^{\alpha-1} \cdot (1-x)^{\beta-1}}{B(\alpha,\beta)}; where\ x [0,1]
\] \[
E[X] = \frac{\alpha}{\alpha + \beta}
\]

``` r
beta <- function(alpha, beta) {
  data = tibble(x = seq(0, 1, 0.001),
                y = dbeta(x, shape1 = alpha, shape2 = beta))
  gf_line(data = data, y ~ x)
}
p1 <- beta(1,2)
p2 <- beta(3,3)
p3 <- beta(20,20)
p4 <- beta(50,10)
p1+p2+p3+p4
```

![](Empirical_Bayes_files/figure-gfm/Plotting%20the%20beta-1.png)<!-- -->

## Batting Averages

In using *Empirical Bayesian Estimation*, we use the available data on
Baseball batting averages to estimate what could be future batting
averages based on new events.

Baseball stuff:  
H: Hits  
AB : “At Bats” ( like an innings in cricket. Maybe)  
Batting Average = H/AB

> A player’s batting average is therefore a percentage between 0 and 1.
> .270 (27%) is considered a typical batting average, while .300 (30%)
> is considered an excellent one.

We have `prior expectations` of players and do not rely on the first few
At Bats to foretell what average a player may attain.

The number of hits a player gets out of his at-bats is a `binomial`
distribution, which models successes out of a total.

``` r
# Preparing `Batting Average` data frame

career <- Batting %>% 
  filter(AB > 0) %>%   # At least one `at bat`
  anti_join(Pitching, by = "playerID") %>%  # leave out the pitchers because they usually are poor batters. No `all-rounders` in baseball! Ha!
  group_by(playerID) %>% 
  summarise(H = sum(H), AB = sum(AB)) %>% 
  mutate(average = H/AB)


# Add Player names please
career <- Master %>% 
  as_tibble() %>% 
  select(playerID, nameFirst, nameLast) %>% 
  unite(name, nameFirst, nameLast) %>% 
  inner_join(career, by = "playerID") %>% 
  select(-playerID)
career
```

    ## # A tibble: 9,670 x 4
    ##    name                  H    AB average
    ##    <chr>             <int> <int>   <dbl>
    ##  1 Hank_Aaron         3771 12364  0.305 
    ##  2 Tommie_Aaron        216   944  0.229 
    ##  3 Andy_Abad             2    21  0.0952
    ##  4 John_Abadie          11    49  0.224 
    ##  5 Ed_Abbaticchio      772  3044  0.254 
    ##  6 Fred_Abbott         107   513  0.209 
    ##  7 Jeff_Abbott         157   596  0.263 
    ##  8 Kurt_Abbott         523  2044  0.256 
    ##  9 Ody_Abbott           13    70  0.186 
    ## 10 Frank_Abercrombie     0     4  0     
    ## # … with 9,660 more rows

``` r
career %>% filter(AB >= 500) %>% gf_histogram(~average,data = .,bins = 50)
```

![](Empirical_Bayes_files/figure-gfm/Estimation%20of%20Priors%20-1-1.png)<!-- -->

> Empirical Bayes is an approximation to more exact Bayesian methods-
> and with the amount of data we have, it’s a very good approximation.
> So far, a beta distribution looks like a pretty appropriate choice
> based on the above histogram. So we know we want to fit the following
> model:

> X ∼ Beta(α0,β0)

> We just need to pick α0 and β0, which we call “hyper-parameters” of
> our model.We can fit these using **maximum likelihood**: to see what
> parameters would maximize the probability of generating the distribu-
> tion we see. There are many functions in R for fitting a probability
> distribution to data (`optim`, `mle`, `bbmle`, etc). You don’t even
> have to use maximum likelihood: you could use the mean and variance,
> called the [“method of
> moments”](https://stats.stackexchange.com/questions/12232/calculating-the-parameters-of-a-beta-distribution-using-the-mean-and-variance).
> We’ll choose to use the `mle` function, and to use dbetabinom.ab.

``` r
#library(stats4)
career_filtered <- career %>% filter(AB>500)
career_filtered
```

    ## # A tibble: 4,209 x 4
    ##    name                H    AB average
    ##    <chr>           <int> <int>   <dbl>
    ##  1 Hank_Aaron       3771 12364   0.305
    ##  2 Tommie_Aaron      216   944   0.229
    ##  3 Ed_Abbaticchio    772  3044   0.254
    ##  4 Fred_Abbott       107   513   0.209
    ##  5 Jeff_Abbott       157   596   0.263
    ##  6 Kurt_Abbott       523  2044   0.256
    ##  7 Brent_Abernathy   212   868   0.244
    ##  8 Shawn_Abner       191   840   0.227
    ##  9 Cal_Abrams        433  1611   0.269
    ## 10 Bobby_Abreu      2470  8480   0.291
    ## # … with 4,199 more rows

``` r
# log-likelihod function
ll <- function(alpha, beta) {
  x <- career_filtered$H
  total <- career_filtered$AB
  - sum(VGAM::dbetabinom.ab(x, total, alpha, beta, log = TRUE))
}
# maximum likelihood estimation
m <- mle(ll, start = list(alpha = 1, beta = 10), method = "L-BFGS-B", lower = c(0.0001, .1))
ab <- coef(m) 
alpha0 <- ab[1]
beta0 <- ab[2]
ab
```

    ## alpha  beta 
    ## 102.0 289.8

``` r
## Method of Moments
estBetaParams <- function(mu, var) {
  alpha <- ((1 - mu) / var - 1 / mu) * mu ^ 2
  beta <- alpha * (1 / mu - 1)
  return(params = list(alpha = alpha, beta = beta))
}
var <- var(career_filtered$average)
mu <- mean(career_filtered$average)
params <- estBetaParams(mu,var)
params
```

    ## $alpha
    ## [1] 79.99
    ## 
    ## $beta
    ## [1] 229.2

``` r
# Checking which estimate is better
career_filtered %>% gf_density(~average, data = .) %>% 
  gf_dist(dist = "beta", kind = "density", params = list(ab[1], ab[2]), color = "red") %>% 
  gf_dist(dist = "beta", kind = "density", params = list(params$alpha, params$beta), color = "blue")
```

![](Empirical_Bayes_files/figure-gfm/Estimation%20of%20Priors%20-2-1.png)<!-- -->

``` r
# Checking using a Kolmogorov-Smirnov test
ks.test(career_filtered$average, "pbeta",ab[1],ab[2])
```

    ## Warning in ks.test(career_filtered$average, "pbeta", ab[1], ab[2]): ties
    ## should not be present for the Kolmogorov-Smirnov test

    ## 
    ##  One-sample Kolmogorov-Smirnov test
    ## 
    ## data:  career_filtered$average
    ## D = 0.048, p-value = 6e-09
    ## alternative hypothesis: two-sided

``` r
ks.test(career_filtered$average, "pbeta",params$alpha,params$beta)
```

    ## Warning in ks.test(career_filtered$average, "pbeta", params$alpha,
    ## params$beta): ties should not be present for the Kolmogorov-Smirnov test

    ## 
    ##  One-sample Kolmogorov-Smirnov test
    ## 
    ## data:  career_filtered$average
    ## D = 0.012, p-value = 0.6
    ## alternative hypothesis: two-sided

``` r
## Not sure how to interpret this test; the lower `p-value` for the `mle` results seems to indicate that those parameters are better. 
```

To follow the book: we continue with the `mle` estimates of the
parameters. Hence the prior distrinution of batting averages is a `beta`
distribution with \(\alpha_0\) = 102.0421 and \(\beta_0\) = 289.7962.

## Updating using Bayesian estimation

Updating the bayesian priors to posteriors is easy when the priors are
beta disctributes, as stated earlier.

The math for proving this is a bit involved (it’s shown
[here](https://en.wikipedia.org/wiki/Conjugate_prior#Example_)), the
result is very simple. The new beta distribution will be:

\[ Beta(α0 + hits, β0 + misses) \] where α0 and β0 are the parameters we
started with- that is, 81 and 219.

If a player has had 300 at-bats and he has hit 100, then the new updated
beta distribution of his average scrores is The math for proving this is
a bit involved (it’s shown here), the result is very simple. The new
beta distribution will be: \[ Beta(α0 + 100, β0 + 200) = Beta(181,419)\]
where α0 and β0 are the parameters we started with- that is, 81 and 219.

If we plot this we get

``` r
#gf_curve(beta(181,419))
```
