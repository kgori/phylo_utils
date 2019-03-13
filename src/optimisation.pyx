# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
import numpy as np
cimport numpy as np
from libc.math cimport fabs

##############################################################
## Parameter transforms for optimisation
##############################################################

cdef int _simplex_encode(double[::1] p, double[::1] theta):
    """
    Convert vector p (length N) to vector theta (length N-1)
    p is constrained to sum to 1, theta is not
    """
    cdef size_t i, N=p.shape[0]
    cdef double y=1.0
    for i in range(N-1):
        theta[i] = p[i] / y
        y -= p[i]
    return 0

def simplex_encode(p):
    theta = np.zeros(p.size - 1)
    _simplex_encode(p, theta)
    return theta

cdef int _simplex_decode(double[::1] theta, double[::1] p):
    cdef size_t i, N=theta.shape[0]
    cdef double x=1.0
    for i in range(N):
        p[i] = theta[i]*x
        x *= 1.0 - theta[i]
    p[N] = x
    return 0

def simplex_decode(theta):
    p = np.zeros(theta.size + 1)
    _simplex_decode(theta, p)
    return p

def logit(p):
    return np.log(p/(1-p))

def expit(p):
    return 1/(1+np.exp(-p))

def transform_params(p):
    return logit(simplex_encode(p))

def decode_params(q):
    return simplex_decode(expit(q))


##############
# OPTIMISATION
##############

cdef int ITMAX = 100
cdef double CGOLD = 0.3819660112501051
cdef double ZEPS=1.0e-10
cdef double TINY = 1e-15


cpdef double quad_interp(double p, double q, double r, double fp, double fq, double fr):
    """
    cpdef double quad_interp(double p, double q, double r, double fp, double fq, double fr)

    Quadratic interpolation of (p, fp), (q, fq), (r, fr) -
    Fits a parabola to the three points and returns the abcissa
    of the turning point of the parabola

    Turning point
    """
    cdef double pfp, pfq, pfr, qfp, qfq, qfr, rfp, rfq, rfr
    cdef double num, div  # Numerator and divisor of extremum point equation E = b/2a
    pfp, pfq, pfr = p*fp, p*fq, p*fr
    qfp, qfq, qfr = q*fp, q*fq, q*fr
    rfp, rfq, rfr = r*fp, r*fq, r*fr
    num = ( -r*rfp + q*qfp - p*pfq + r*rfq - q*qfr + p*pfr )
    div = (qfp - rfp + rfq - pfq + pfr - qfr)
    if div < 0 and -div < TINY:
        div = -TINY
    elif div >= 0 and div < TINY:
        div = TINY
    extremum = num / (2 * div)
    return extremum

cdef void brent(double ax, double bx, double cx, f, double tol, double[:] out):
    """
    brent(double ax, double bx, double cx, f, double tol, double[:] out)

    Given a function f, and given a bracketing triplet of abscissas ax, bx, cx (such that bx is
    between ax and cx, and f(bx) is less than both f(ax) and f(cx)), this routine isolates
    the minimum to a fractional precision of about tol using Brent's method. The out array is
    filled with the x, f(x) and the number of iterations.
    """

    cdef int iter
    cdef double a, b, d, etemp, fu, fv, fw, fx, p, q, r, tol1, tol2, u, v, w, x, xm
    cdef double e = 0.0

    a, b = (ax, cx) if (ax < cx) else (cx, ax)

    x = w = v = bx
    fw = fv = fx = f(x)
    for iter in xrange(1, ITMAX+1): # Main loop
        xm = 0.5*(a + b)
        tol1 = tol*fabs(x) + ZEPS
        tol2 = 2.0*tol1
        if (fabs(x - xm) <= (tol2 - 0.5*(b - a))): # Test for done here.
            out[0] = x
            out[1] = fx
            out[2] = iter
            return

        if (fabs(e) > tol1):  # Construct a trial parabolic fit
            r = (x - w)*(fx - fv)
            q = (x - v)*(fx - fw)
            p = (x - v)*q - (x - w)*r
            q = 2.0*(q - r)
            if (q > 0.0):
                p = -p

            q = fabs(q)
            etemp = e
            e = d
            if fabs(p) >= fabs(0.5*q*etemp) or p <= q*(a - x) or p >= q*(b - x):
                e = (a-x) if (x >= xm) else (b-x)
                d = CGOLD*e
            # The above conditions determine the acceptability of the parabolic fit. Here we
            # take the golden section step into the larger of the two segments.
            else:
                d = p / q  # Take the parabolic step
                u = x + d
                if u - a < tol2 or b - u < tol2:
                    d = tol1 if (xm-x) >=0 else -tol1


        else:
            e = (a-x) if (x >= xm) else (b-x)
            d = CGOLD*e

        if fabs(d) >= tol1:
            u = x+d
        else:
            u = x + (tol1 if d >= 0 else -tol1)

        fu = f(u) # This is the one function evaluation per iteration

        if (fu <= fx):   # Now decide what to do with our
            if (u >= x): # function evaluation
                a = x
            else:
                b = x
            v, w, x = w, x, u  # Housekeeping follows:
            fv, fw, fx = fw, fx, fu

        else:
            if (u < x):
                a = u
            else:
                b = u
            if (fu <= fw or w == x):
                v = w
                w = u
                fv = fw
                fw = fu

            elif (fu <= fv or v == x or v == w):
                v = u
                fv = fu
            # Done with housekeeping. Back for
            # another iteration.

    out[0] = x
    out[1] = f(x)
    out[2] = ITMAX+1
    return


cpdef void dbrent(double ax, double bx, double cx, f, df, double tol, double[:] out):
    """
    Given a function f and its derivative function df, and given a bracketing triplet of abscissas ax,
    bx, cx [such that bx is between ax and cx, and f(bx) is less than both f(ax) and f(cx)],
    this routine isolates the minimum to a fractional precision of about tol using a modification of
    Brent's method that uses derivatives. The abscissa of the minimum is returned as xmin, and
    the minimum function value is returned as dbrent, the returned function value.
    """

    assert out.size >= 3
    cdef int iter,ok1,ok2 # Will be used as flags for whether proposed steps are acceptable or not.
    cdef double a, b, d, d1, d2, du, dv, dw, dx, e=0.0
    cdef double fu, fv, fw, fx, olde, tol1, tol2, u, u1, u2, v, w, x, xm

    # Comments following will point out only differences from the routine brent. Read that
    # routine first.
    a, b = (ax, cx) if (ax < cx) else (cx, ax)

    x = w = v = bx  # All our housekeeping chores are doubled
    fw=fv=fx=f(x)   # by the necessity of moving
    dw=dv=dx=df(x)  # derivative values around as well
                    # as function values.

    for iter in range(1, ITMAX):
        xm = 0.5*(a+b)
        tol1 = tol*fabs(x)+ZEPS
        tol2 = 2.0*tol1
        if fabs(x-xm) <= (tol2-0.5*(b-a)):
            out[0] = x
            out[1] = fx
            out[2] = iter
            return

        if fabs(e) > tol1:
            d1=2.0*(b-a) # Initialize these d’s to an out-of-bracket
            d2=d1        # value.

            if dw != dx:
                d1=(w-x)*dx/(dx-dw) # Secant method with one point.
            if dv != dx:
                d2=(v-x)*dx/(dx-dv) # And the other.

            # Which of these two estimates of d shall we take? We will insist that they be within
            # the bracket, and on the side pointed to by the derivative at x:

            u1=x+d1;
            u2=x+d2;

            ok1 = (a-u1)*(u1-b) > 0.0 and dx*d1 <= 0.0;
            ok2 = (a-u2)*(u2-b) > 0.0 and dx*d2 <= 0.0;
            olde=e # Movement on the step before last.
            e=d
            if ok1 or ok2: # Take only an acceptable d, and if
                           # both are acceptable, then take
                           # the smallest one.

                if ok1 and ok2:
                    d = d1 if fabs(d1) < fabs(d2) else d2

                elif ok1:
                    d=d1

                else:
                    d=d2
                if fabs(d) <= fabs(0.5*olde):
                    u=x+d
                    if u-a < tol2 or b-u < tol2:
                        d= tol1 if (xm-x) >= 0 else -tol1

                else: # Bisect, not golden section.
                    e = a-x if dx >= 0 else b-x
                    d=0.5*e
                    # Decide which segment by the sign of the derivative.

            else:
                e = a-x if dx >= 0.0 else b-x
                d=0.5*e

        else:
            e = a-x if dx >= 0.0 else b-x
            d = 0.5*e

        if fabs(d) >= tol1:
            u = x + d
            fu = f(u)
        else:
            u = x + (tol if d >=0 else -tol)
            fu = f(u)
            if fu > fx: # If the minimum step in the downhill
                        # direction takes us uphill, then
                        # we are done.
                out[0] = x
                out[1] = fx
                out[2] = iter
                return

        du=df(u) # Now all the housekeeping, sigh.
        if (fu <= fx):
            if (u >= x):
                a=x
            else:
                b=x;
            v,fv,dv = w,fw,dw
            w,fw,dw = x,fx,dx
            x,fx,dx = u,fu,du
        else:
            if (u < x):
                a=u
            else:
                b=u
            if (fu <= fw or w == x):
                v,fv,dv = w,fw,dw
                w,fw,dw = u,fu,du
            elif (fu < fv or v == x or v == w):
                v,fv,dv = u,fu,du

    out[0] = x
    out[1] = f(x)
    out[2] = ITMAX+1


def dbrent_wrap(double guess, double lbracket, double rbracket, fn, dfn, tol=1.5e-8):
    """
    dbrent_wrap(double guess, double lbracket, double rbracket, fn, dfn, tol=1.5e-8)
    """
    out = np.zeros(3, dtype=np.double)
    dbrent(lbracket, rbracket, guess, fn, dfn, tol, out)
    return out

def brent_wrap(double guess, double lbracket, double rbracket, fn, tol=1.5e-8):
    """
    brent_wrap(double guess, double lbracket, double rbracket, fn, tol=1.5e-8)
    """
    out = np.zeros(3, dtype=np.double)
    brent(lbracket, rbracket, guess, fn, tol, out)
    return out