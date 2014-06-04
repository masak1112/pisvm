#include "svm_solver_nu.h"
#include <stdio.h>

void Solver_NU::Solve(int l, const QMatrix& Q, const double *b, const schar *y,
	     double *alpha, double Cp, double Cn, double eps,
	     SolutionInfo* si, int shrinking)
{
  this->si = si;
  Solver::Solve(l,Q,b,y,alpha,Cp,Cn,eps,si,shrinking);
}

int Solver_NU::select_working_set(int &out_i, int &out_j)
{
  // Always select the maximal violating pair. Old fashion.
  // Does the same as LibSVM v2.36.
  double Gmin1 = INF; double Gmin2 = INF;
  double Gmax1 = -INF; double Gmax2 = -INF;
  int min1 = -1; int min2 = -1;
  int max1 = -1; int max2 = -1;
//   printf("G = ");
//   for(int t=0; t<l; ++t)
//     printf(" %g",G[t]);
//   printf("\n");
  for(int t=0; t<active_size; ++t)
    {
      if(y[t] == +1)
	{
	  if(!is_upper_bound(t))
	    {
	      if(G[t] < Gmin1)
		{
		  Gmin1 = G[t];
		  min1 = t;
		}
	    }
	  if(!is_lower_bound(t))
	    {
	      if(G[t] > Gmax1)
		{
		  Gmax1 = G[t];
		  max1 = t;
		}
	    }
	}
      else
	{
	  if(!is_upper_bound(t))
	    {
	      if(G[t] < Gmin2)
		{
		  Gmin2 = G[t];
		  min2 = t;
		}
	    }
	  if(!is_lower_bound(t))
	    {
	      if(G[t] > Gmax2)
		{
		  Gmax2 = G[t];
		  max2 = t;
		}
	    }
	}
    }
  if(max(Gmax1-Gmin1,Gmax2-Gmin2) < eps)
    return 1;
  if(Gmax1-Gmin1 > Gmax2-Gmin2)
    {
      out_i = max1;
      out_j = min1;
    }
  else
    {
      out_i = max2;
      out_j = min2;
    }
//   printf("Selected (%d,%d)\n",out_i,out_j);
  return 0;
}

// return 1 if already optimal, return 0 otherwise
// int Solver_NU::select_working_set(int &out_i, int &out_j)
// {
//   // return i,j such that y_i = y_j and
//   // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
//   // j: minimizes the decrease of obj value
//   //    (if quadratic coefficeint <= 0, replace it with tau)
//   //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

//   double Gmaxp = -INF;
//   int Gmaxp_idx = -1;

//   double Gmaxn = -INF;
//   int Gmaxn_idx = -1;

//   int Gmin_idx = -1;
//   double obj_diff_min = INF;

//   for(int t=0;t<active_size;t++)
//     if(y[t]==+1)
//       {
// 	if(!is_upper_bound(t))
// 	  if(-G[t] >= Gmaxp)
// 	    {
// 	      Gmaxp = -G[t];
// 	      Gmaxp_idx = t;
// 	    }
//       }
//     else
//       {
// 	if(!is_lower_bound(t))
// 	  if(G[t] >= Gmaxn)
// 	    {
// 	      Gmaxn = G[t];
// 	      Gmaxn_idx = t;
// 	    }
//       }

//   int ip = Gmaxp_idx;
//   int in = Gmaxn_idx;
//   const Qfloat *Q_ip = NULL;
//   const Qfloat *Q_in = NULL;
//   if(ip != -1) // NULL Q_ip not accessed: Gmaxp=-INF if ip=-1
//     Q_ip = Q->get_Q(ip,active_size);
//   if(in != -1)
//     Q_in = Q->get_Q(in,active_size);

//   for(int j=0;j<active_size;j++)
//     {
//       if(y[j]==+1)
// 	{
// 	  if (!is_lower_bound(j))	
// 	    {
// 	      double grad_diff=Gmaxp+G[j];
// 	      if (grad_diff >= eps)
// 		{
// 		  double obj_diff; 
// 		  double quad_coef = Q_ip[ip]+QD[j]-2*Q_ip[j];
// 		  if (quad_coef > 0)
// 		    obj_diff = -(grad_diff*grad_diff)/quad_coef;
// 		  else
// 		    obj_diff = -(grad_diff*grad_diff)/TAU;

// 		  if (obj_diff <= obj_diff_min)
// 		    {
// 		      Gmin_idx=j;
// 		      obj_diff_min = obj_diff;
// 		    }
// 		}
// 	    }
// 	}
//       else
// 	{
// 	  if (!is_upper_bound(j))
// 	    {
// 	      double grad_diff=Gmaxn-G[j];
// 	      if (grad_diff >= eps)
// 		{
// 		  double obj_diff; 
// 		  double quad_coef = Q_in[in]+QD[j]-2*Q_in[j];
// 		  if (quad_coef > 0)
// 		    obj_diff = -(grad_diff*grad_diff)/quad_coef;
// 		  else
// 		    obj_diff = -(grad_diff*grad_diff)/TAU;

// 		  if (obj_diff <= obj_diff_min)
// 		    {
// 		      Gmin_idx=j;
// 		      obj_diff_min = obj_diff;
// 		    }
// 		}
// 	    }
// 	}
//     }

//   if(Gmin_idx == -1)
//     return 1;

//   if (y[Gmin_idx] == +1)
//     out_i = Gmaxp_idx;
//   else
//     out_i = Gmaxn_idx;
//   out_j = Gmin_idx;
//   return 0;
// }

void Solver_NU::do_shrinking()
{
  double Gmax1 = -INF;	// max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
  double Gmax2 = -INF;	// max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
  double Gmax3 = -INF;	// max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
  double Gmax4 = -INF;	// max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

  // find maximal violating pair first
  int k;
  for(k=0;k<active_size;k++)
    {
      if(!is_upper_bound(k))
	{
	  if(y[k]==+1)
	    {
	      if(-G[k] > Gmax1) Gmax1 = -G[k];
	    }
	  else	if(-G[k] > Gmax3) Gmax3 = -G[k];
	}
      if(!is_lower_bound(k))
	{
	  if(y[k]==+1)
	    {	
	      if(G[k] > Gmax2) Gmax2 = G[k];
	    }
	  else	if(G[k] > Gmax4) Gmax4 = G[k];
	}
    }

  // shrinking

  double Gm1 = -Gmax2;
  double Gm2 = -Gmax1;
  double Gm3 = -Gmax4;
  double Gm4 = -Gmax3;

  for(k=0;k<active_size;k++)
    {
      if(is_lower_bound(k))
	{
	  if(y[k]==+1)
	    {
	      if(-G[k] >= Gm1) continue;
	    }
	  else	if(-G[k] >= Gm3) continue;
	}
      else if(is_upper_bound(k))
	{
	  if(y[k]==+1)
	    {
	      if(G[k] >= Gm2) continue;
	    }
	  else	if(G[k] >= Gm4) continue;
	}
      else continue;

      --active_size;
      swap_index(k,active_size);
      --k;	// look at the newcomer
    }

  // unshrink, check all variables again before final iterations

  if(unshrinked || max(-(Gm1+Gm2),-(Gm3+Gm4)) > eps*10) return;
	
  unshrinked = true;
  reconstruct_gradient();

  for(k=l-1;k>=active_size;k--)
    {
      if(is_lower_bound(k))
	{
	  if(y[k]==+1)
	    {
	      if(-G[k] < Gm1) continue;
	    }
	  else	if(-G[k] < Gm3) continue;
	}
      else if(is_upper_bound(k))
	{
	  if(y[k]==+1)
	    {
	      if(G[k] < Gm2) continue;
	    }
	  else	if(G[k] < Gm4) continue;
	}
      else continue;

      swap_index(k,active_size);
      active_size++;
      ++k;	// look at the newcomer
    }
}

double Solver_NU::calculate_rho()
{
  int nr_free1 = 0,nr_free2 = 0;
  double ub1 = INF, ub2 = INF;
  double lb1 = -INF, lb2 = -INF;
  double sum_free1 = 0, sum_free2 = 0;

//   printf("alpha = ");
//   for(int i=0; i<l; ++i)
//     printf(" %g",alpha[i]);
//   printf("\n");

  for(int i=0;i<active_size;i++)
    {
      if(y[i]==+1)
	{
	  if(is_lower_bound(i))
	    ub1 = min(ub1,G[i]);
	  else if(is_upper_bound(i))
	    lb1 = max(lb1,G[i]);
	  else
	    {
	      ++nr_free1;
	      sum_free1 += G[i];
	    }
	}
      else
	{
	  if(is_lower_bound(i))
	    ub2 = min(ub2,G[i]);
	  else if(is_upper_bound(i))
	    lb2 = max(lb2,G[i]);
	  else
	    {
	      ++nr_free2;
	      sum_free2 += G[i];
	    }
	}
    }
  printf("nr_free1 = %d\n", nr_free1);
  printf("sum_free1 = %g\n",sum_free1);
  printf("nr_free2 = %d\n", nr_free2);
  printf("sum_freee = %g\n",sum_free2);
  double r1,r2;
  if(nr_free1 > 0)
    r1 = sum_free1/nr_free1;
  else
    r1 = (ub1+lb1)/2;
	
  if(nr_free2 > 0)
    r2 = sum_free2/nr_free2;
  else
    r2 = (ub2+lb2)/2;
	
  si->r = (r1+r2)/2;
  printf("(r1+r2)/2 = %g\n", (r1+r2)/2);
  printf("(r1+r2)/2 = %g\n", (r1-r2)/2);
  return (r1-r2)/2;
}
