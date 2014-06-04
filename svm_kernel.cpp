#include "svm_kernel.h"

Kernel::Kernel(int l, Xfloat **x_, int **nz_idx_,
               const int *x_len_, const int max_idx_,
               const svm_parameter& param)
    :kernel_type(param.kernel_type), degree(param.degree),
     gamma(param.gamma), coef0(param.coef0)
{
    switch(kernel_type)
    {
    case LINEAR:
        kernel_function = &Kernel::kernel_linear;
        break;
    case POLY:
        kernel_function = &Kernel::kernel_poly;
        break;
    case RBF:
        kernel_function = &Kernel::kernel_rbf;
        break;
    case SIGMOID:
        kernel_function = &Kernel::kernel_sigmoid;
        break;
    }

    clone(x,x_,l);
    clone(nz_idx,nz_idx_,l);
    clone(x_len,x_len_,l);
    max_idx = max_idx_;
    v = new Xfloat[max_idx];
    unrolled = 0;
    for(int k=0; k<x_len[unrolled]; ++k)
        v[nz_idx[unrolled][k]] = x[unrolled][k];

    if(kernel_type == RBF)
    {
        x_square = new double[l];
        for(int i=0; i<l; i++)
            x_square[i] = dot(i,i);
    }
    else
        x_square = 0;
}

Kernel::~Kernel()
{
    delete[] x;
    delete[] nz_idx;
    delete[] x_len;
    delete[] v;
    delete[] x_square;
}

double Kernel::dot(const Xfloat *x, const int *nz_x, const int lx,
                   const Xfloat *y, const int *nz_y, const int ly)
{
    register double sum = 0;
    register int i = 0;
    register int j = 0;
    while(i < lx && j < ly)
    {
        if(nz_x[i] == nz_y[j])
        {
            sum += x[i] * y[j];
            ++i;
            ++j;
        }
        else if(nz_x[i] > nz_y[j])
            ++j;
        else if(nz_x[i] < nz_y[j])
            ++i;
    }
    return sum;
}

double Kernel::k_function(const Xfloat *x, const int *nz_x, const int lx,
                          Xfloat *y, int *nz_y, int ly,
                          const svm_parameter& param)
{
    switch(param.kernel_type)
    {
    case LINEAR:
        return dot(x, nz_x, lx, y, nz_y, ly);
    case POLY:
        return powi(param.gamma*dot(x, nz_x, lx, y, nz_y, ly)
                    +param.coef0,param.degree);
    case RBF:
    {
        return exp(-param.gamma*(
                       dot(x, nz_x, lx, x, nz_x, lx)+
                       dot(y, nz_y, ly, y, nz_y, ly)-
                       2*dot(x, nz_x, lx, y, nz_y, ly)));
    }
    case SIGMOID:
        return tanh(param.gamma*dot(x, nz_x, lx, y, nz_y, ly)
                    +param.coef0);
    default:
        return 0;	/* Unreachable */
    }
}
