#ifndef SVM_KERNEL_INCLUDED
#define SVM_KERNEL_INCLUDED
#include <math.h>
#include "svm.h"
#include "util.h"
//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//

inline double powi(double base, int times)
{
  double tmp = base, ret = 1.0;
  for(int t=times; t>0; t/=2)
    {
      if(t%2==1)
	ret *= tmp;
      tmp = tmp * tmp;
    }
  return ret;
}

class QMatrix {
public:
  virtual Qfloat *get_Q(int column, int len) const = 0;
  virtual Qfloat *get_QD() const = 0;
  virtual Qfloat *get_Q_subset(int i, int *idxs, int n) const = 0;
  virtual Qfloat get_non_cached(int i, int j) const = 0;
  virtual bool is_cached(int i) const = 0;
  virtual void swap_index(int i, int j) const = 0;
  virtual ~QMatrix() {}
};  

class Kernel: public QMatrix {
public:
  Kernel(int l, Xfloat **x, int **nz_idx, 
	 const int *x_len, const int max_idx, 
	 const svm_parameter& param);
  virtual ~Kernel();

  static double k_function(const Xfloat *x, const int *nz_x, const int lx,
			   Xfloat *y, int *nz_y, int ly, 
			   const svm_parameter& param);
  virtual Qfloat *get_Q(int column, int len) const = 0;
  virtual Qfloat *get_QD() const = 0;
  virtual Qfloat *get_Q_subset(int i, int *idxs, int n) const = 0;
  virtual Qfloat get_non_cached(int i, int j) const = 0;
  virtual bool is_cached(int i) const = 0;
  virtual void swap_index(int i, int j) const	// no so const...
  {
    swap(x[i],x[j]);
    swap(nz_idx[i], nz_idx[j]);
    swap(x_len[i], x_len[j]);
    if(unrolled == i)
      unrolled = j;
    else if(unrolled == j)
      unrolled = i;
    if(x_square) swap(x_square[i],x_square[j]);
  }
protected:

  double (Kernel::*kernel_function)(int i, int j) const;

private:
  Xfloat **x;
  int **nz_idx;
  int *x_len;
  double *x_square;
  // dense unrolled sparse vector
  mutable Xfloat *v; 
  // index of currently unrolled vector
  mutable int unrolled;
  int max_idx;

  // svm_parameter
  const int kernel_type;
  const int degree;
  const double gamma;
  const double coef0;

  static double dot(const Xfloat *x, const int *nz_x, const int lx, 
		    const Xfloat *y, const int *nz_y, const int ly);
  double dot(const int i, const int j) const
  {
    register int k;
    register double sum;
    if(i != unrolled)
      {
	for(k=0; k<x_len[unrolled]; ++k)
	  v[nz_idx[unrolled][k]] = 0;
	unrolled = i;
	for(k=0; k<x_len[i]; ++k)
	  v[nz_idx[i][k]] = x[i][k];
      }
    sum = 0;
    for(k=0; k<x_len[j]; ++k)
      sum += v[nz_idx[j][k]] * x[j][k];
    return sum;
  }
  double kernel_linear(int i, int j) const
  {
    return dot(i,j);
  }
  double kernel_poly(int i, int j) const
  {
    return powi(gamma*dot(i,j)+coef0,degree);
  }
  double kernel_rbf(int i, int j) const
  {
    return exp(-gamma*(x_square[i]+x_square[j]-2*dot(i,j)));
  }
  double kernel_sigmoid(int i, int j) const
  {
    return tanh(gamma*dot(i,j)+coef0);
  }
};
#endif
