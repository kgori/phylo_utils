#include <stdio.h>
void LnlCalc(double *probs, double *partials, double *return_value, int states, int sites) {
    double entry;
    // 2darray[i, j] = 1darray[i*ncol+j] -> here 'states' has role of ncol
    for (int i=0; i < sites; ++i) {
        for (int j=0; j < states; ++j) {
            entry=0;
            for (int k=0; k < states; ++k) {
                entry += probs[states*j+k] * partials[states*i+k];
            }
            return_value[states*i+j] = entry;
        }
    }
}
