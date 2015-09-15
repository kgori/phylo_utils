#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "discrete_gamma.h"
// PAML code for discrete gamma

// Forward decs
double QuantileChi2(double prob, double v);
double LnGammaFunction(double alpha);
double IncompleteGamma(double x, double alpha, double ln_gamma_alpha);

#define QuantileGamma(prob, alpha, beta)                                       \
    QuantileChi2(prob, 2.0 * (alpha)) / (2.0 * (beta))

long factorial(int n) {
    long f = 1, i;
    if (n > 11)
        return -1;
    for (i = 2; i <= (long)n; i++)
        f *= i;
    return (f);
}

/* functions concerning the CDF and percentage points of the gamma and
   Chi2 distribution
*/
double QuantileNormal(double prob) {
    /* returns z so that Prob{x<z}=prob where x ~ N(0,1) and
       (1e-12)<prob<1-(1e-12)
       returns (-9999) if in error
       Odeh RE & Evans JO (1974) The percentage points of the normal
       distribution.
       Applied Statistics 22: 96-97 (AS70)

       Newer methods:
         Wichura MJ (1988) Algorithm AS 241: the percentage points of the
           normal distribution.  37: 477-484.
         Beasley JD & Springer SG  (1977).  Algorithm AS 111: the percentage
           points of the normal distribution.  26: 118-121.
    */
    double a0 = -.322232431088, a1 = -1, a2 = -.342242088547,
           a3 = -.0204231210245;
    double a4 = -.453642210148e-4, b0 = .0993484626060, b1 = .588581570495;
    double b2 = .531103462366, b3 = .103537752850, b4 = .0038560700634;
    double y, z = 0, p = prob, p1;

    p1 = (p < 0.5 ? p : 1 - p);
    if (p1 < 1e-20)
        z = 999;
    else {
        y = sqrt(log(1 / (p1 * p1)));
        z = y +
            ((((y * a4 + a3) * y + a2) * y + a1) * y + a0) /
                ((((y * b4 + b3) * y + b2) * y + b1) * y + b0);
    }
    return (p < 0.5 ? -z : z);
}

double LnGammaFunction(double alpha) {
    /* returns ln(gamma(alpha)) for alpha>0, accurate to 10 decimal places.
       Stirling's formula is used for the central polynomial part of the
       procedure.
       Pike MC & Hill ID (1966) Algorithm 291: Logarithm of the gamma function.
       Communications of the Association for Computing Machinery, 9:684
    */
    double x = alpha, f = 0, z;

    if (x < 7) {
        f = 1;
        z = x - 1;
        while (++z < 7)
            f *= z;
        x = z;
        f = -log(f);
    }
    z = 1 / (x * x);
    return f + (x - 0.5) * log(x) - x + .918938533204673 +
           (((-.000595238095238 * z + .000793650793651) * z -
             .002777777777778) *
                z +
            .083333333333333) /
               x;
}

double LnGamma(double x) {
    /* returns ln(gamma(x)) for x>0, accurate to 10 decimal places.
       Stirling's formula is used for the central polynomial part of the
       procedure.

       Pike MC & Hill ID (1966) Algorithm 291: Logarithm of the gamma function.
       Communications of the Association for Computing Machinery, 9:684
    */
    double f = 0, fneg = 0, z, lng;
    int nx = (int)x;

    if ((double)nx == x && nx >= 0 && nx <= 11)
        lng = log((double)factorial(nx - 1));
    else {
        if (x <= 0) {
            printf("LnGamma(%.6f) not implemented", x);
            if ((int)x - x == 0) {
                puts("lnGamma undefined");
                return (-1);
            }
            for (fneg = 1; x < 0; x++)
                fneg /= x;
            if (fneg < 0)
                return (-1);
            fneg = log(fneg);
        }
        if (x < 7) {
            f = 1;
            z = x - 1;
            while (++z < 7)
                f *= z;
            x = z;
            f = -log(f);
        }
        z = 1 / (x * x);
        lng = fneg + f + (x - 0.5) * log(x) - x + .918938533204673 +
              (((-.000595238095238 * z + .000793650793651) * z -
                .002777777777778) *
                   z +
               .083333333333333) /
                  x;
    }
    return lng;
}

double QuantileChi2(double prob, double v) {
    /* returns z so that Prob{x<z}=prob where x is Chi2 distributed with df=v
       returns -1 if in error.   0.000002<prob<0.999998
       RATNEST FORTRAN by
           Best DJ & Roberts DE (1975) The percentage Quantiles of the
           Chi2 distribution.  Applied Statistics 24: 385-388.  (AS91)
       Converted into C by Ziheng Yang, Oct. 1993.
    */
    double e = .5e-6, aa = .6931471805, p = prob, g;
    double xx, c, ch, a = 0, q = 0, p1 = 0, p2 = 0, t = 0, x = 0, b = 0, s1, s2,
                      s3, s4, s5, s6;

    if (p < .000002 || p > .999998 || v <= 0)
        return (-1);

    g = LnGammaFunction(v / 2);
    xx = v / 2;
    c = xx - 1;
    if (v >= -1.24 * log(p))
        goto l1;

    ch = pow((p * xx * exp(g + xx * aa)), 1 / xx);
    if (ch - e < 0)
        return (ch);
    goto l4;
l1:
    if (v > .32)
        goto l3;
    ch = 0.4;
    a = log(1 - p);
l2:
    q = ch;
    p1 = 1 + ch * (4.67 + ch);
    p2 = ch * (6.73 + ch * (6.66 + ch));
    t = -0.5 + (4.67 + 2 * ch) / p1 - (6.73 + ch * (13.32 + 3 * ch)) / p2;
    ch -= (1 - exp(a + g + .5 * ch + c * aa) * p2 / p1) / t;
    if (fabs(q / ch - 1) - .01 <= 0)
        goto l4;
    else
        goto l2;

l3:
    x = QuantileNormal(p);
    p1 = 0.222222 / v;
    ch = v * pow((x * sqrt(p1) + 1 - p1), 3.0);
    if (ch > 2.2 * v + 6)
        ch = -2 * (log(1 - p) - c * log(.5 * ch) + g);
l4:
    q = ch;
    p1 = .5 * ch;
    if ((t = IncompleteGamma(p1, xx, g)) < 0) {
        printf("\nerr IncompleteGamma");
        return (-1);
    }
    p2 = p - t;
    t = p2 * exp(xx * aa + g + p1 - c * log(ch));
    b = t / ch;
    a = 0.5 * t - b * c;

    s1 = (210 + a * (140 + a * (105 + a * (84 + a * (70 + 60 * a))))) / 420;
    s2 = (420 + a * (735 + a * (966 + a * (1141 + 1278 * a)))) / 2520;
    s3 = (210 + a * (462 + a * (707 + 932 * a))) / 2520;
    s4 = (252 + a * (672 + 1182 * a) + c * (294 + a * (889 + 1740 * a))) / 5040;
    s5 = (84 + 264 * a + c * (175 + 606 * a)) / 2520;
    s6 = (120 + c * (346 + 127 * c)) / 5040;
    ch +=
        t * (1 + 0.5 * t * s1 -
             b * c * (s1 - b * (s2 - b * (s3 - b * (s4 - b * (s5 - b * s6))))));
    if (fabs(q / ch - 1) > e)
        goto l4;

    return (ch);
}

double IncompleteGamma(double x, double alpha, double ln_gamma_alpha) {
    /* returns the incomplete gamma ratio I(x,alpha) where x is the upper
               limit of the integration and alpha is the shape parameter.
       returns (-1) if in error
       ln_gamma_alpha = ln(Gamma(alpha)), is almost redundant.
       (1) series expansion     if (alpha>x || x<=1)
       (2) continued fraction   otherwise
       RATNEST FORTRAN by
       Bhattacharjee GP (1970) The incomplete gamma integral.  Applied
       Statistics,
       19: 285-287 (AS32)
    */
    int i;
    double p = alpha, g = ln_gamma_alpha;
    double accurate = 1e-8, overflow = 1e30;
    double factor, gin = 0, rn = 0, a = 0, b = 0, an = 0, dif = 0, term = 0,
                   pn[6];

    if (x == 0)
        return (0);
    if (x < 0 || p <= 0)
        return (-1);

    factor = exp(p * log(x) - x - g);
    if (x > 1 && x >= p)
        goto l30;
    /* (1) series expansion */
    gin = 1;
    term = 1;
    rn = p;
l20:
    rn++;
    term *= x / rn;
    gin += term;

    if (term > accurate)
        goto l20;
    gin *= factor / p;
    goto l50;
l30:
    /* (2) continued fraction */
    a = 1 - p;
    b = a + x + 1;
    term = 0;
    pn[0] = 1;
    pn[1] = x;
    pn[2] = x + 1;
    pn[3] = x * b;
    gin = pn[2] / pn[3];
l32:
    a++;
    b += 2;
    term++;
    an = a * term;
    for (i = 0; i < 2; i++)
        pn[i + 4] = b * pn[i + 2] - an * pn[i];
    if (pn[5] == 0)
        goto l35;
    rn = pn[4] / pn[5];
    dif = fabs(gin - rn);
    if (dif > accurate)
        goto l34;
    if (dif <= accurate * rn)
        goto l42;
l34:
    gin = rn;
l35:
    for (i = 0; i < 4; i++)
        pn[i] = pn[i + 2];
    if (fabs(pn[4]) < overflow)
        goto l32;
    for (i = 0; i < 4; i++)
        pn[i] /= overflow;
    goto l32;
l42:
    gin = 1 - factor * gin;

l50:
    return (gin);
}

int DiscreteGamma(double freqK[], double rK[], double alpha, double beta, int K,
                  int UseMedian) {
    /* discretization of G(alpha, beta) with equal proportions in each category.
    */
    int i;
    double t, mean = alpha / beta, lnga1;

    if (UseMedian) { /* median */
        for (i = 0; i < K; i++) {
            rK[i] = QuantileGamma((i * 2. + 1) / (2. * K), alpha, beta);
        }
        for (i = 0, t = 0; i < K; i++) {
            t += rK[i];
        }
        for (i = 0; i < K; i++) {
            rK[i] *= mean * K / t; /* rescale so that the mean is alpha/beta. */
        }
    } else {                       /* mean */
        lnga1 = LnGamma(alpha + 1);
        for (i = 0; i < K - 1; i++) { /* cutting points, Eq. 9 */
            freqK[i] = QuantileGamma((i + 1.0) / K, alpha, beta);
        }
        for (i = 0; i < K - 1; i++) { /* Eq. 10 */
            freqK[i] = IncompleteGamma(freqK[i] * beta, alpha + 1, lnga1);
        }
        rK[0] = freqK[0] * mean * K;
        for (i = 1; i < K - 1; i++) {
            rK[i] = (freqK[i] - freqK[i - 1]) * mean * K;
        }
        rK[K - 1] = (1 - freqK[K - 2]) * mean * K;
    }

    for (i = 0; i < K; i++)
        freqK[i] = 1.0 / K;

    return (0);
}
