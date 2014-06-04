// D. Brugger, september 2006
// parallel_loqo.c - parallel implementation of LOQO qp solver using PLAPACK.
// $Id: parallel_loqo.c 573 2010-12-29 10:54:20Z dome $
//
// Copyright (C) 2006 Dominik Brugger
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#include <PLA.h>
#include "PLA_missing_decls.h"
#include "parallel_loqo.h"

#define PREDICTOR 1
#define CORRECTOR 2

#define	max(A, B)	((A) > (B) ? (A) : (B))
#define	ABS(A)  	((A) > 0 ? (A) : (-(A)))
#define sqr(A)          ((A) * (A))

#ifndef CheckError
#define CheckError(n) if(n){printf("line %d, file %s, error %d\n",__LINE__,__FILE__,n); }
#endif

/* Auxiliary functions for setting and getting diagonal values of a 
   matrix.
*/

void PLA_Set_unit_diagonal(PLA_Obj *a)
{
  PLA_Obj a_cur = NULL, a11 = NULL; 
  void *a11_buf;
  int size, info;
  int local_n, local_m;
  info = PLA_Obj_view_all(*a, &a_cur); CheckError(info);
  while(1)
    {
      info = PLA_Obj_global_length(a_cur, &size); CheckError(info);
      if(size == 0) break;
      info = PLA_Obj_split_4(a_cur, 1, 1, &a11, PLA_DUMMY, PLA_DUMMY, &a_cur);
      CheckError(info);
      info = PLA_Obj_local_length(a11, &local_m); CheckError(info);
      info = PLA_Obj_local_width(a11, &local_n); CheckError(info);
      if(local_n == 1 && local_m == 1)
	{
	  info = PLA_Obj_local_buffer(a11, (void **) &a11_buf); 
	  CheckError(info);
	  *((LOQOfloat*) a11_buf) = 1;
	}
    }
}

void print_matrix(PLA_Obj A)
{
  void *local_buf = NULL;
  int local_m, local_n, local_ldim;
  int i,j;
  PLA_Obj_local_length( A, &local_m );
  PLA_Obj_local_width(  A, &local_n );
  PLA_Obj_local_buffer( A, (void **) &local_buf );
  PLA_Obj_local_ldim(   A, &local_ldim );
  for (j=0; j<local_n; j++ )
    {
      for (i=0; i<local_m; i++ )
	printf(" %g", ((LOQOfloat *) local_buf)[ j*local_ldim + i ]);
      printf("\n");
    }
  fflush(stdout);
}

void print_vector(PLA_Obj v)
{
  void *addr_buf = NULL;
  int info;
  int local_stride, local_n, i;
  info = PLA_Obj_local_length(v, &local_n); CheckError(info);
  info = PLA_Obj_local_stride(v, &local_stride); CheckError(info);  
  info = PLA_Obj_local_buffer(v, (void **) &addr_buf); CheckError(info);
  for(i=0; i<local_n; ++i)
    {
      printf(" %g", ((LOQOfloat *) addr_buf)[local_stride*i]);
    }
  printf("\n"); fflush(stdout);
}

/* Assuming vectors have same length + layout */
void PLA_Copy_vector(PLA_Obj from, PLA_Obj to)
{
  PLA_Template templ;
  MPI_Comm comm;
  void *from_buf = NULL, *to_buf = NULL;
  int info;
  int local_stride, local_n, i;
  info = PLA_Obj_template(from, &templ); CheckError(info);
  info = PLA_Temp_comm_all(templ, &comm); CheckError(info);
  info = PLA_Obj_local_length(from, &local_n); CheckError(info);
  info = PLA_Obj_local_stride(from, &local_stride); CheckError(info);  
  info = PLA_Obj_local_buffer(from, (void **) &from_buf); CheckError(info);
  info = PLA_Obj_local_buffer(to, (void **) &to_buf); CheckError(info);
  for(i=0; i<local_n; ++i)
    {
      ((LOQOfloat*) to_buf)[i*local_stride] = ((LOQOfloat *) from_buf)[i*local_stride];
    }
  info = MPI_Barrier(comm); CheckError(info);
}

void PLA_Shift_diagonal(PLA_Obj *a, LOQOfloat shift)
{
  PLA_Obj a_cur = NULL, a11 = NULL; 
  void *a11_buf;
  int size, info;
  int local_m, local_n;
  info = PLA_Obj_view_all(*a, &a_cur);
  while(1)
    {
      info = PLA_Obj_global_length(a_cur, &size);
      if(size == 0) break;
      info = PLA_Obj_split_4(a_cur, 1, 1, &a11, PLA_DUMMY, PLA_DUMMY, &a_cur);
      info = PLA_Obj_local_length(a11, &local_m); CheckError(info);
      info = PLA_Obj_local_width(a11, &local_n); CheckError(info);
      if(local_m == 1 && local_n == 1)
	{
	  info = PLA_Obj_local_buffer(a11, (void **) &a11_buf);
	  *((LOQOfloat*) a11_buf) += shift;
	}
    }
}

void PLA_Set_diagonal(PLA_Obj *a, PLA_Obj d)
{
  PLA_Obj a_cur = NULL, d_cur = NULL, a11 = NULL, d1 = NULL; 
  void *a11_buf; void *d1_buf;
  int size, info;
  int local_m, local_n, local_m2;
  info = PLA_Obj_view_all(*a, &a_cur);
  info = PLA_Obj_view_all(d, &d_cur);
  while(1)
    {
      info = PLA_Obj_global_length(a_cur, &size);
      if(size == 0) break;
      info = PLA_Obj_split_4(a_cur, 1, 1, &a11, PLA_DUMMY, PLA_DUMMY, &a_cur);
      info = PLA_Obj_horz_split_2(d_cur, 1, &d1, &d_cur);
      info = PLA_Obj_local_length(a11, &local_m); CheckError(info);
      info = PLA_Obj_local_width(a11, &local_n); CheckError(info);
      info = PLA_Obj_local_length(d1, &local_m2); CheckError(info);
      if(local_m == 1 && local_n == 1 && local_m2 == 1)
	{
	  info = PLA_Obj_local_buffer(a11, (void **) &a11_buf);
	  info = PLA_Obj_local_buffer(d1, (void **) &d1_buf);
	  *((LOQOfloat*) a11_buf) = *((LOQOfloat*) d1_buf);
	}
    }
}

void PLA_Get_diagonal(PLA_Obj a, PLA_Obj *d)
{
  PLA_Obj a_cur = NULL, d_cur = NULL, a11 = NULL, d1 = NULL; 
  void *a11_buf; void *d1_buf;
  int size, info;
  int local_m, local_n, local_m2;
  info = PLA_Obj_view_all(a, &a_cur);
  info = PLA_Obj_view_all(*d, &d_cur);
  while(1)
    {
      info = PLA_Obj_global_length(a_cur, &size);
      if(size == 0) break;
      info = PLA_Obj_split_4(a_cur, 1, 1, &a11, PLA_DUMMY, PLA_DUMMY, &a_cur);
      info = PLA_Obj_horz_split_2(d_cur, 1, &d1, &d_cur);
      info = PLA_Obj_local_length(a11, &local_m); CheckError(info);
      info = PLA_Obj_local_width(a11, &local_n); CheckError(info);
      info = PLA_Obj_local_length(d1, &local_m2); CheckError(info);
      if(local_m == 1 && local_n == 1 && local_m2 == 1)
	{
	  info = PLA_Obj_local_buffer(a11, (void **) &a11_buf);
	  info = PLA_Obj_local_buffer(d1, (void **) &d1_buf);
	  *((LOQOfloat*) d1_buf) = *((LOQOfloat*) a11_buf);
	}
    }
}

/* Cholesky decomposition with manteuffel shifting. */
int PLA_Choldc(PLA_Obj a)
{
  int shifted, i, idx;
  int info = 0;
  LOQOfloat rs;
  LOQOfloat shift_amount = 0.0;
  LOQOfloat one = 1.0;
  LOQOfloat *array = NULL;
  int error,parameters,sequential,r12;
  do{
    shifted = 0;
    /* This is necessary as PLA_Chol, returns the index
       of a non-positive pivot in info only if error checking
       is disabled. */
    info = PLA_Get_error_checking(&error, &parameters, &sequential, &r12);
    info = PLA_Set_error_checking(FALSE, FALSE, FALSE, FALSE);
    info = PLA_Chol(PLA_LOWER_TRIANGULAR, a);
    if(info != 0){
      /* We need a zero based index */
      idx = info-1;
      info = PLA_API_begin(); CheckError(info);
      info = PLA_Obj_API_open(a); CheckError(info);
      array = (LOQOfloat*) malloc((idx+1)*sizeof(LOQOfloat));
      /* Set array to zero, because of axpy op. */
      for(i=0; i<idx+1; ++i)
	array[i] = 0;
      info = PLA_API_axpy_global_to_matrix(1,idx+1,&one,a,idx,0,
					   (void *)array,1); CheckError(info);
      info = PLA_Obj_API_close(a); CheckError(info);
      info = PLA_API_end(); CheckError(info);
      /* Compute part of row sum. */
      rs = 0.0;
      for(i=0; i<idx; ++i) /* Add pivot value below */
	rs += fabs(array[i]);
      free(array);

      info = PLA_API_begin(); CheckError(info);
      info = PLA_Obj_API_open(a); CheckError(info);
      array = (LOQOfloat*) malloc((idx+1)*sizeof(LOQOfloat));
      /* Set array to zero, because of axpy op. */
      for(i=0; i<idx+1; ++i)
	array[i] = 0;
      info = PLA_API_axpy_global_to_matrix(idx+1,1,&one,a,idx,idx,
					   (void *)array,1); CheckError(info);
      info = PLA_Obj_API_close(a); CheckError(info);
      info = PLA_API_end(); CheckError(info);
      /* Complete computation of row sum. */
      for(i=0; i<idx+1; ++i)
	rs += fabs(array[i]);
      shift_amount = max(rs,1.1*shift_amount);
      printf("using shift_amount = %g\n", shift_amount); fflush(stdout);
      free(array);

      PLA_Shift_diagonal(&a, shift_amount);
      shifted = 1;
    }
    /* Restore error checking state. */
    info = PLA_Set_error_checking(error, parameters, sequential, r12);
  } while(shifted);

  return info;
}

/* Solves reduced KKT system:
   | -Q1 A^T | | delta_alpha | = | c1 |
   | A   Q2  | | delta_h     |   | c2 |
   Dimension of Q1 is n x n. Dimension of A is m x n.
   Y1T and stores intermediary result computed during predictor step.
   Dimension of Y1T is m x n.
*/
void parallel_solve_reduced(PLA_Obj Q1, PLA_Obj Q2, PLA_Obj A, PLA_Obj c1, PLA_Obj c2, 
			    PLA_Obj *delta_alpha, PLA_Obj *delta_h, PLA_Obj *Y1T,
			    int step)
{
  int info;
  PLA_Obj Y2 = NULL, minus_one = NULL, plus_one = NULL;
  PLA_Template templ = NULL;
  info = PLA_Obj_template(Q1, &templ); CheckError(info);
  info = PLA_Mvector_create_conf_to(*delta_alpha, 1, &Y2); CheckError(info);
  info = PLA_Mscalar_create(MPIfloat, PLA_ALL_ROWS, PLA_ALL_COLS, 1, 1, 
			    templ, &plus_one); CheckError(info);
  info = PLA_Obj_set_to_one(plus_one); CheckError(info);
  info = PLA_Mscalar_create(MPIfloat, PLA_ALL_ROWS, PLA_ALL_COLS, 1, 1, 
			    templ, &minus_one); CheckError(info);
  info = PLA_Obj_set_to_minus_one(minus_one); CheckError(info);
  if(step == PREDICTOR)
    {
      /* Compute cholesky decomposition: Q1 <- L1*L1^T
         overwriting lower triangular part of Q1 */
/*       info = PLA_Chol(PLA_LOWER_TRIANGULAR, Q1); CheckError(info); */
      info = PLA_Choldc(Q1); CheckError(info);
      /* Compute: Y1^T <- A * (L1^-1)^T.
	 Dimension of Y1^T is m x n. */
      info = PLA_Copy(A, *Y1T); CheckError(info);
      info = PLA_Trsm(PLA_SIDE_RIGHT, PLA_LOWER_TRIANGULAR, PLA_TRANS,
		      PLA_NONUNIT_DIAG, plus_one, Q1, *Y1T); CheckError(info);
      /* Compute cholesky decomposition: (Q2 + Y1^T*Y1) <- L2*L2^T. */
      info = PLA_Syrk(PLA_LOWER_TRIANGULAR, PLA_NO_TRANS, plus_one, *Y1T,
		      plus_one, Q2); CheckError(info);
/*       info = PLA_Chol(PLA_LOWER_TRIANGULAR, Q2); CheckError(info);  */
      info = PLA_Choldc(Q2); CheckError(info);
    }
  /* Compute Y2 <- L1^-1*c1. 
     Dimension of Y2 is n x 1. */
  PLA_Copy_vector(c1, Y2);
  info = PLA_Trsv(PLA_LOWER_TRIANGULAR, PLA_NO_TRANS, PLA_NONUNIT_DIAG,
		  Q1, Y2); CheckError(info);
  /* Compute delta_h <- c2 + Y1^T*Y2. */
  PLA_Copy_vector(c2, *delta_h);
  info = PLA_Gemv(PLA_NO_TRANS, plus_one, *Y1T, Y2, plus_one, *delta_h);
  CheckError(info);
  /* Compute delta_h <- L2^-1*(c2 + Y1^T*Y2). */
  info = PLA_Trsv(PLA_LOWER_TRIANGULAR, PLA_NO_TRANS, PLA_NONUNIT_DIAG,
		  Q2, *delta_h); CheckError(info);
  /* Finally compute delta_h <- L2^-T * (L2^-1*(c2 + Y1^T*Y2). */
  info = PLA_Trsv(PLA_LOWER_TRIANGULAR, PLA_TRANS, PLA_NONUNIT_DIAG,
		  Q2, *delta_h); CheckError(info);
  /* Compute delta_alpha <- Y1*delta_h - Y2. */
  PLA_Copy_vector(Y2, *delta_alpha);
  info = PLA_Gemv(PLA_TRANS, plus_one, *Y1T, *delta_h, minus_one, 
		  *delta_alpha); CheckError(info);
  /* Compute delta_alpha <- L1^-T * (Y1*delta_h - Y2). */
  info = PLA_Trsv(PLA_LOWER_TRIANGULAR, PLA_TRANS, PLA_NONUNIT_DIAG, Q1,
		  *delta_alpha); CheckError(info);
  info = PLA_Obj_free(&Y2); CheckError(info);
  info = PLA_Obj_free(&plus_one); CheckError(info);
  info = PLA_Obj_free(&minus_one); CheckError(info);
}

/* Struct for local buffer access. */
struct local_buffer_address{
  LOQOfloat *c;
  LOQOfloat *b;

  LOQOfloat *x;
  LOQOfloat *l;
  LOQOfloat *u;
  LOQOfloat *g;
  LOQOfloat *t;
  LOQOfloat *s;
  LOQOfloat *z;

  LOQOfloat *delta_x;
  LOQOfloat *delta_g;
  LOQOfloat *delta_t;
  LOQOfloat *delta_z;
  LOQOfloat *delta_s;

  LOQOfloat *sigma;
  LOQOfloat *nu;
  LOQOfloat *tau;
  LOQOfloat *hat_nu;
  LOQOfloat *hat_tau;
  LOQOfloat *gamma_z;
  LOQOfloat *gamma_s;

  LOQOfloat *c_plus_1;
  LOQOfloat *b_plus_1;
  LOQOfloat *primal_inf;
  LOQOfloat *dual_inf;
  LOQOfloat *primal_obj;
  LOQOfloat *dual_obj;
  LOQOfloat *x_h_x;
  LOQOfloat *tmp;
  LOQOfloat *mu;
  LOQOfloat *alfa;
  LOQOfloat *max_ratio;

  LOQOfloat *d;
  LOQOfloat *c_x;
  LOQOfloat *c_y;
};
   
int parallel_loqo(PLA_Obj c, PLA_Obj h_x, PLA_Obj a, PLA_Obj b, PLA_Obj l,
		  PLA_Obj u, PLA_Obj *x, PLA_Obj *g, PLA_Obj *t, PLA_Obj *y,
		  PLA_Obj *z, PLA_Obj *s, PLA_Obj *dist, int verb, LOQOfloat sigfig_max, 
		  int counter_max, LOQOfloat margin, LOQOfloat bound, int restart)
{
  /* To be allocated. */
  PLA_Obj diag_h_x = NULL, h_y = NULL, y1t = NULL, c_x = NULL, c_y = NULL; 
  PLA_Obj h_dot_x = NULL, rho = NULL, nu = NULL, tau = NULL, sigma = NULL;
  PLA_Obj gamma_z = NULL, gamma_s = NULL, hat_nu = NULL, hat_tau = NULL;
  PLA_Obj delta_x = NULL, delta_y = NULL, delta_s = NULL, delta_z = NULL;
  PLA_Obj delta_g = NULL, delta_t = NULL, d = NULL, ones_n = NULL;
  PLA_Obj ones_m = NULL, max_ratio = NULL;
  PLA_Obj h_x_copy = NULL;
  
  /* Multiscalars */
  PLA_Obj b_plus_1 = NULL, c_plus_1 = NULL, x_h_x = NULL, primal_inf = NULL;
  PLA_Obj dual_inf = NULL, primal_obj = NULL, dual_obj = NULL, mu = NULL;
  PLA_Obj alfa = NULL, plus_one = NULL, minus_one = NULL, tmp = NULL;
  PLA_Obj max_idx = NULL;

  /* Local buffer stuff */
  void *addr_buf;
  struct local_buffer_address loc;
  int idx, local_n, local_stride, local_m, local_stride2;

  /* Instrumentation etc. */
  LOQOfloat sigfig = 0.0;
  LOQOfloat next_ratio = 0.0;
  int counter = 0;
  int status = STILL_RUNNING;
  int i,n,m,info;
  PLA_Template templ = NULL;
  MPI_Comm comm;
  int rank, numnodes;

  info = PLA_Set_error_checking(FALSE,FALSE,FALSE,FALSE);
  CheckError(info);

  /* Extract template and MPI information */
  info = PLA_Obj_template(h_x, &templ); CheckError(info);
  info = PLA_Temp_comm_all_info(templ, &comm, &rank, &numnodes); 
  CheckError(info);
  info = PLA_Obj_global_length(h_x, &n); CheckError(info);
  info = PLA_Obj_global_length(b, &m); CheckError(info);

/*   print_vector(c); */
/*   print_matrix(h_x); */
/*   print_matrix(a); */
/*   print_vector(b); */
/*   print_vector(l); */
/*   print_vector(u); */

/*   printf("Allocating objs..."); fflush(stdout); */
  /* Allocation */
  info = PLA_Mvector_create_conf_to(c, 1, &diag_h_x); CheckError(info);
  info = PLA_Matrix_create(MPIfloat, m, m, templ, PLA_ALIGN_FIRST, 
			   PLA_ALIGN_FIRST, &h_y); CheckError(info);
  info = PLA_Matrix_create(MPIfloat, m, n, templ, PLA_ALIGN_FIRST,
			   PLA_ALIGN_FIRST, &y1t); CheckError(info);
  info = PLA_Matrix_create(MPIfloat, n, n, templ, PLA_ALIGN_FIRST,
			   PLA_ALIGN_FIRST, &h_x_copy); CheckError(info);
  info = PLA_Mvector_create_conf_to(c, 1, &c_x); CheckError(info);
  info = PLA_Mvector_create_conf_to(b, 1, &c_y); CheckError(info);
  info = PLA_Mvector_create_conf_to(c, 1, &h_dot_x); CheckError(info);
  info = PLA_Mvector_create_conf_to(b, 1, &rho); CheckError(info);
  info = PLA_Mvector_create_conf_to(c, 1, &nu); CheckError(info);
  info = PLA_Mvector_create_conf_to(c, 1, &tau); CheckError(info);
  info = PLA_Mvector_create_conf_to(c, 1, &sigma); CheckError(info);
  info = PLA_Mvector_create_conf_to(c, 1, &gamma_z); CheckError(info);
  info = PLA_Mvector_create_conf_to(c, 1, &gamma_s); CheckError(info);
  info = PLA_Mvector_create_conf_to(c, 1, &hat_nu); CheckError(info);
  info = PLA_Mvector_create_conf_to(c, 1, &hat_tau); CheckError(info);
  info = PLA_Mvector_create_conf_to(c, 1, &delta_x); CheckError(info);
  info = PLA_Mvector_create_conf_to(b, 1, &delta_y); CheckError(info);
  info = PLA_Mvector_create_conf_to(c, 1, &delta_s); CheckError(info);
  info = PLA_Mvector_create_conf_to(c, 1, &delta_z); CheckError(info);
  info = PLA_Mvector_create_conf_to(c, 1, &delta_g); CheckError(info);
  info = PLA_Mvector_create_conf_to(c, 1, &delta_t); CheckError(info);
  info = PLA_Mvector_create_conf_to(c, 1, &d); CheckError(info);
  info = PLA_Mvector_create_conf_to(c, 1, &ones_n); CheckError(info);
  info = PLA_Mvector_create_conf_to(b, 1, &ones_m); CheckError(info);
  info = PLA_Mvector_create_conf_to(c, 1, &max_ratio); CheckError(info);

  info = PLA_Mscalar_create(MPIfloat, PLA_ALL_ROWS, PLA_ALL_COLS, 1, 1, 
			    templ, &b_plus_1); CheckError(info);
  info = PLA_Mscalar_create(MPIfloat, PLA_ALL_ROWS, PLA_ALL_COLS, 1, 1, 
			    templ, &c_plus_1); CheckError(info);
  info = PLA_Mscalar_create(MPIfloat, PLA_ALL_ROWS, PLA_ALL_COLS, 1, 1, 
			    templ, &x_h_x); CheckError(info);
  info = PLA_Mscalar_create(MPIfloat, PLA_ALL_ROWS, PLA_ALL_COLS, 1, 1, 
			    templ, &primal_inf); CheckError(info);
  info = PLA_Mscalar_create(MPIfloat, PLA_ALL_ROWS, PLA_ALL_COLS, 1, 1, 
			    templ, &dual_inf); CheckError(info);
  info = PLA_Mscalar_create(MPIfloat, PLA_ALL_ROWS, PLA_ALL_COLS, 1, 1, 
			    templ, &primal_obj); CheckError(info);
  info = PLA_Mscalar_create(MPIfloat, PLA_ALL_ROWS, PLA_ALL_COLS, 1, 1, 
			    templ, &dual_obj); CheckError(info);
  info = PLA_Mscalar_create(MPIfloat, PLA_ALL_ROWS, PLA_ALL_COLS, 1, 1, 
			    templ, &mu); CheckError(info);
  info = PLA_Mscalar_create(MPIfloat, PLA_ALL_ROWS, PLA_ALL_COLS, 1, 1, 
			    templ, &alfa); CheckError(info);
  info = PLA_Mscalar_create(MPIfloat, PLA_ALL_ROWS, PLA_ALL_COLS, 1, 1, 
			    templ, &plus_one); CheckError(info);
  info = PLA_Mscalar_create(MPIfloat, PLA_ALL_ROWS, PLA_ALL_COLS, 1, 1, 
			    templ, &minus_one); CheckError(info);
  info = PLA_Mscalar_create(MPIfloat, PLA_ALL_ROWS, PLA_ALL_COLS, 1, 1, 
			    templ, &tmp); CheckError(info);
  info = PLA_Mscalar_create(MPI_INT, PLA_ALL_ROWS, PLA_ALL_COLS, 1, 1, 
			    templ, &max_idx); CheckError(info);
/*   printf("done.\n"); fflush(stdout); */
  /* Initialize pointers to local buffers. */
  loc.c = NULL;
  info = PLA_Obj_local_buffer(c, (void **) &addr_buf);
  loc.c = (LOQOfloat *) addr_buf;
  loc.b = NULL;
  info = PLA_Obj_local_buffer(b, (void **) &addr_buf);
  loc.b = (LOQOfloat *) addr_buf;
  loc.x = NULL;
  info = PLA_Obj_local_buffer(*x, (void **) &addr_buf);
  loc.x = (LOQOfloat *) addr_buf;
  loc.l = NULL;
  info = PLA_Obj_local_buffer(l, (void **) &addr_buf);
  loc.l = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(u, (void **) &addr_buf);
  loc.u = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(*g, (void **) &addr_buf);
  loc.g = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(*t, (void **) &addr_buf);
  loc.t = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(*s, (void **) &addr_buf);
  loc.s = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(*z, (void **) &addr_buf);
  loc.z = (LOQOfloat *) addr_buf;

  info = PLA_Obj_local_buffer(delta_x, (void **) &addr_buf);
  loc.delta_x = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(delta_g, (void **) &addr_buf);
  loc.delta_g = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(delta_t, (void **) &addr_buf);
  loc.delta_t = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(delta_z, (void **) &addr_buf);
  loc.delta_z = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(delta_s, (void **) &addr_buf);
  loc.delta_s = (LOQOfloat *) addr_buf;

  info = PLA_Obj_local_buffer(sigma, (void **) &addr_buf);
  loc.sigma = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(nu, (void **) &addr_buf);
  loc.nu = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(tau, (void **) &addr_buf);
  loc.tau = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(hat_nu, (void **) &addr_buf);
  loc.hat_nu = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(hat_tau, (void **) &addr_buf);
  loc.hat_tau = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(gamma_z, (void **) &addr_buf);
  loc.gamma_z = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(gamma_s, (void **) &addr_buf);
  loc.gamma_s = (LOQOfloat *) addr_buf;

  info = PLA_Obj_local_buffer(c_plus_1, (void **) &addr_buf);
  loc.c_plus_1 = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(b_plus_1, (void **) &addr_buf);
  loc.b_plus_1 = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(primal_inf, (void **) &addr_buf);
  loc.primal_inf = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(dual_inf, (void **) &addr_buf);
  loc.dual_inf = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(primal_obj, (void **) &addr_buf);
  loc.primal_obj = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(dual_obj, (void **) &addr_buf);
  loc.dual_obj = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(x_h_x, (void **) &addr_buf);
  loc.x_h_x = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(tmp, (void **) &addr_buf);
  loc.tmp = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(mu, (void **) &addr_buf);
  loc.mu = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(alfa, (void **) &addr_buf);
  loc.alfa = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(max_ratio, (void **) &addr_buf);
  loc.max_ratio = (LOQOfloat *) addr_buf;

  info = PLA_Obj_local_buffer(d, (void **) &addr_buf);
  loc.d = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(c_x, (void **) &addr_buf);
  loc.c_x = (LOQOfloat *) addr_buf;
  info = PLA_Obj_local_buffer(c_y, (void **) &addr_buf);
  loc.c_y = (LOQOfloat *) addr_buf;

  /* Determine local size of vectors and stride. */
  info = PLA_Obj_local_length(*x, &local_n); CheckError(info);
  info = PLA_Obj_local_stride(*x, &local_stride); CheckError(info);
  info = PLA_Obj_local_length(b, &local_m); CheckError(info);
  info = PLA_Obj_local_stride(b, &local_stride2); CheckError(info);

  /* Initial settings */
/*   printf("Initial settings ..."); fflush(stdout); */
  info = PLA_Obj_set_to_one(plus_one); CheckError(info);
  info = PLA_Obj_set_to_minus_one(minus_one); CheckError(info);
  info = PLA_Obj_set_to_one(ones_n); CheckError(info);
  info = PLA_Obj_set_to_one(ones_m); CheckError(info);
  info = PLA_Dot(c, c, c_plus_1); CheckError(info);
  info = PLA_Dot(b, b, b_plus_1); CheckError(info);
/*   info = PLA_Dot(c, ones_n, c_plus_1); CheckError(info); */
/*   info = PLA_Dot(b, ones_m, b_plus_1); CheckError(info); */
  *loc.c_plus_1 = sqrt(*loc.c_plus_1);
  *loc.b_plus_1 = sqrt(*loc.b_plus_1);
  *loc.c_plus_1 += 1.0;
  *loc.b_plus_1 += 1.0;
/*   printf("done.\n"); fflush(stdout); */

  /* Get diagonal terms */
/*   printf("Get diagonal terms..."); fflush(stdout); */
  PLA_Get_diagonal(h_x, &diag_h_x);
  PLA_Copy(h_x, h_x_copy);
/*   printf("done.\n"); fflush(stdout); */

/*   printf("Setting initial values..."); */
  /* Starting point */
  if(restart == 1)
    {
      for(i=0; i<local_n; ++i)
	{
	  idx = i*local_stride;
	  loc.g[idx] = max(ABS(loc.x[idx] - loc.l[idx]), bound);
	  loc.t[idx] = max(ABS(loc.u[idx] - loc.x[idx]), bound);
	}
      /* Compute h_dot_x <- h_x * x */
/*       PLA_Copy_vector(*x, h_dot_x); */
      info = PLA_Obj_set_to_zero(h_dot_x); CheckError(info);
/*       info = PLA_Symv(PLA_UPPER_TRIANGULAR, plus_one, h_x, *x, plus_one, h_dot_x);       */
      info = PLA_Gemv(PLA_NO_TRANS, plus_one, h_x_copy, *x, plus_one, 
		      h_dot_x); CheckError(info);      
      /* Compute sigma <- c + h_dot_x - A^T*y */
      PLA_Copy_vector(h_dot_x, sigma);
      info = PLA_Gemv(PLA_TRANS, minus_one, a, *y, plus_one, sigma);
      CheckError(info);
      info = PLA_Axpy(plus_one, c, sigma); CheckError(info);
      /* MPI Barrier to wait for changes in sigma. */
      info = MPI_Barrier(comm);
      /* Set s and z */
      for(i=0; i<local_n; ++i)
	{
	  idx = i*local_stride;
	  if(loc.sigma[idx] > 0)
	    {
	      loc.s[idx] = bound;
	      loc.z[idx] = loc.sigma[idx] + bound;
	    }
	  else
	    {
	      loc.s[idx] = bound - loc.sigma[idx];
	      loc.z[idx] = bound;
	    }
	}
    } /* if(restart == 1) */
  else
    {
      /* Use default start settings */

      /* Set h_y <- m x m identity matrix */
      info = PLA_Obj_set_to_zero(h_y); CheckError(info);
      PLA_Set_unit_diagonal(&h_y);
      /* Set h_x <- h_x + n x n identity matrix */
      info = PLA_Obj_set_to_one(d); CheckError(info);
      info = PLA_Axpy(plus_one, diag_h_x, d); CheckError(info);
      PLA_Set_diagonal(&h_x, d);
      /* Set c_x <- c and c_y <- b */
      PLA_Copy_vector(c, c_x);
      PLA_Copy_vector(b, c_y);
      /* Solve the reduced KKT system. */
      parallel_solve_reduced(h_x, h_y, a, c_x, c_y, x, y, &y1t, PREDICTOR);
      /* MPI Barrier to wait for changes in x. */
      info = MPI_Barrier(comm); CheckError(info);

      /* Initialize the other variables besides x and y. */
      for(i=0; i<local_n; ++i)
	{
	  idx = i*local_stride;
	  loc.g[idx] = max(ABS(loc.x[idx] - loc.l[idx]), bound);
	  loc.z[idx] = max(ABS(loc.x[idx]), bound);
	  loc.t[idx] = max(ABS(loc.u[idx] - loc.x[idx]), bound);
	  loc.s[idx] = max(ABS(loc.x[idx]), bound);
	}
    }
  /* MPI Barrier to wait for local changes in s,t,z and g. */
  info = MPI_Barrier(comm); CheckError(info);

  /* Set mu <- (s^T * t + z^T * g) / (2*n) */
  info = PLA_Dot(*z, *g, mu); CheckError(info);
  info = PLA_Dot(*s, *t, tmp); CheckError(info);
  info = MPI_Barrier(comm);
  *loc.mu = (*loc.mu + (*loc.tmp)) / (2*n);
/*   printf("Setting initial values done.\n"); fflush(stdout); */

  /* The main loop. */
  if(verb >= STATUS && rank == 0)
    {
      printf("counter | pri_inf  | dual_inf  | pri_obj   | dual_obj  | ");
      printf("sigfig | alpha  | nu \n");
      printf("-------------------------------------------------------");
      printf("---------------------------\n");
    }
  while(status == STILL_RUNNING)
    {
      /* Predictor step. */
      /* Put back original diagonal values. */
      PLA_Set_diagonal(&h_x, diag_h_x);

      /* Set h_dot_x <- h_x * x */
/*       PLA_Copy_vector(*x, h_dot_x); */
      info = PLA_Obj_set_to_zero(h_dot_x); CheckError(info);
/*        info = PLA_Symv(PLA_UPPER_TRIANGULAR, plus_one, h_x, *x, plus_one, h_dot_x);  */
      info = PLA_Gemv(PLA_NO_TRANS, plus_one, h_x_copy, *x, plus_one, 
		      h_dot_x); CheckError(info);
      /* Set rho <- b - A * x */
      PLA_Copy_vector(b, rho);
      info = PLA_Gemv(PLA_NO_TRANS, minus_one, a, *x, plus_one, rho); 
      CheckError(info);
      /* Set nu <- l - x + g */
      PLA_Copy_vector(l, nu);
      info = PLA_Axpy(minus_one, *x, nu); CheckError(info);
      info = PLA_Axpy(plus_one, *g, nu); CheckError(info);
      /* Set tau <- u - x - t */
      PLA_Copy_vector(u, tau);
      info = PLA_Axpy(minus_one, *x, tau); CheckError(info);
      info = PLA_Axpy(minus_one, *t, tau); CheckError(info);
      /* Set sigma <- c - z + s + h_dot_x - A^T * y */
      PLA_Copy_vector(c, sigma);
      info = PLA_Axpy(minus_one, *z, sigma); CheckError(info);
      info = PLA_Axpy(plus_one, *s, sigma); CheckError(info);
      info = PLA_Axpy(plus_one, h_dot_x, sigma); CheckError(info);
      info = PLA_Gemv(PLA_TRANS, minus_one, a, *y, plus_one, sigma);
      /* Set gamma_z <- -z, gamma_s <- -s. */
      info = PLA_Obj_set_to_zero(gamma_s); CheckError(info);
      info = PLA_Axpy(minus_one, *s, gamma_s); CheckError(info);
      info = PLA_Obj_set_to_zero(gamma_z); CheckError(info);
      info = PLA_Axpy(minus_one, *z, gamma_z); CheckError(info);
/*       print_vector(h_dot_x); */
/*       print_vector(rho); */
/*       print_vector(nu); */
/*       print_vector(tau); */
/*       print_vector(sigma); */
      /* Instrumentation */
      info = PLA_Dot(h_dot_x, *x, x_h_x); CheckError(info);
      {
	/* Compute 
	   primal_inf <- sqrt(tau^T * tau + nu^T * nu + rho^T * rho) 
	   / b_plus_1. */
	info = PLA_Dot(tau, tau, primal_inf); CheckError(info);
	info = PLA_Dot(nu, nu, tmp); CheckError(info);
	info = MPI_Barrier(comm);
	*loc.primal_inf += *loc.tmp;
	info = PLA_Dot(rho, rho, tmp); CheckError(info);
	info = MPI_Barrier(comm);
	*loc.primal_inf += *loc.tmp;
	*loc.primal_inf = sqrt(*loc.primal_inf)/(*loc.b_plus_1);
      }
      {
	/* Compute 
	   dual_inf <- sqrt(sigma^T * sigma) / c_plus_1. */
	info = PLA_Dot(sigma, sigma, dual_inf); CheckError(info);
	info = MPI_Barrier(comm);
	*loc.dual_inf = sqrt(*loc.dual_inf)/(*loc.c_plus_1);

      }
      {
	/* Compute
	   primal_obj <- 0.5 * x_h_x + c^T * x. */
	info = PLA_Dot(c, *x, primal_obj); CheckError(info);
	info = MPI_Barrier(comm);
	*loc.primal_obj += 0.5*(*loc.x_h_x);
      }
      {
	/* Compute
	   dual_obj <- -0.5 * x_h_x + l^T * z - u^T * s + b^T * y. */
	info = PLA_Dot(l, *z, dual_obj); CheckError(info);
	info = PLA_Dot(b, *y, tmp); CheckError(info);
	info = MPI_Barrier(comm);
	*loc.dual_obj += *loc.tmp;
	info = PLA_Dot(u, *s, tmp); CheckError(info);
	info = MPI_Barrier(comm);
	*loc.dual_obj -= *loc.tmp;
	*loc.dual_obj -= 0.5*(*loc.x_h_x);
      }
      sigfig = log10(ABS(*loc.primal_obj) + 1) -
	log10(ABS(*loc.primal_obj - (*loc.dual_obj)));
      sigfig = max(sigfig, 0);

      /* The diagnostics - after we computed our results we will
	 analyze them */
      if (counter > counter_max) status = ITERATION_LIMIT;
      if (sigfig  > sigfig_max)  status = OPTIMAL_SOLUTION;
      if (*loc.primal_inf > 10e100)   status = PRIMAL_INFEASIBLE;
      if (*loc.dual_inf > 10e100)     status = DUAL_INFEASIBLE;
      if (*loc.primal_inf > 10e100 && *loc.dual_inf > 10e100) 
	status = PRIMAL_AND_DUAL_INFEASIBLE;
      if (ABS(*loc.primal_obj) > 10e100) status = PRIMAL_UNBOUNDED;
      if (ABS(*loc.dual_obj) > 10e100) status = DUAL_UNBOUNDED;
      
      /* Generate report */
      if ((verb >= FLOOD) | ((verb == STATUS) & (status != 0)) && rank == 0)
	printf("%7i | %.2e | %.2e | % .2e | % .2e | %6.3f | %.4f | %.2e\n",
	       counter, *loc.primal_inf, *loc.dual_inf, *loc.primal_obj, 
	       *loc.dual_obj, sigfig, *loc.alfa, *loc.mu);

      counter++;
      if(status == 0){
	/* Set intermediary hat variables. */
	for(i=0; i<local_n; ++i)
	  {
	    idx = i*local_stride;
	    loc.hat_nu[idx] = loc.nu[idx] + loc.g[idx] * 
	      loc.gamma_z[idx] / loc.z[idx];
	    loc.hat_tau[idx] = loc.tau[idx] - loc.t[idx] *
	      loc.gamma_s[idx] / loc.s[idx];
	    loc.d[idx] = loc.z[idx] / loc.g[idx] + loc.s[idx] / loc.t[idx];
	    loc.c_x[idx] = loc.sigma[idx] - loc.z[idx] * loc.hat_nu[idx] /
	      loc.g[idx] - loc.s[idx] * loc.hat_tau[idx] / loc.t[idx];
	  }
	info = MPI_Barrier(comm); CheckError(info);

	/* Set h_x <- h_x + diag(z^-1 * g + s * t^-1). */
	info = PLA_Copy(h_x_copy, h_x); CheckError(info);
	info = PLA_Axpy(plus_one, diag_h_x, d); CheckError(info);
	PLA_Set_diagonal(&h_x, d);
	/* Set c_y <- rho. */
	PLA_Copy_vector(rho, c_y);
	/* Set h_y <-  0. */
	info = PLA_Obj_set_to_zero(h_y); CheckError(info);
	
	/* Compute predictor step */
	parallel_solve_reduced(h_x, h_y, a, c_x, c_y, &delta_x, &delta_y, &y1t, 
			       PREDICTOR);
	info = MPI_Barrier(comm); CheckError(info);

	/* Do backsubstitution */
	for(i=0; i<local_n; ++i)
	  {
	    idx = i*local_stride;
	    loc.delta_s[idx] = loc.s[idx] * 
	      (loc.delta_x[idx] - loc.hat_tau[idx]) / loc.t[idx];
	    loc.delta_z[idx] = loc.z[idx] *
	      (loc.hat_nu[idx] - loc.delta_x[idx]) / loc.g[idx];

	    loc.delta_g[idx] = loc.g[idx] *
	      (loc.gamma_z[idx] - loc.delta_z[idx]) / loc.z[idx];
	    loc.delta_t[idx] = loc.t[idx] *
	      (loc.gamma_s[idx] - loc.delta_s[idx]) / loc.s[idx];
	    
	    /* Central path (corrector) */
	    loc.gamma_z[idx] = *loc.mu / loc.g[idx] - loc.z[idx] -
	      loc.delta_z[idx] * loc.delta_g[idx] / loc.g[idx];
	    loc.gamma_s[idx] = *loc.mu / loc.t[idx] - loc.s[idx] -
	      loc.delta_s[idx] * loc.delta_t[idx] / loc.t[idx];
	    
	    /* The hat variables. */
	    loc.hat_nu[idx] = loc.nu[idx] + loc.g[idx] * 
	      loc.gamma_z[idx] / loc.z[idx];
	    loc.hat_tau[idx] = loc.tau[idx] - loc.t[idx] *
	      loc.gamma_s[idx] / loc.s[idx];
	    
	    loc.c_x[idx] = loc.sigma[idx] - loc.z[idx] * loc.hat_nu[idx] /
	      loc.g[idx] - loc.s[idx] * loc.hat_tau[idx] / loc.t[idx];
	  }
	info = MPI_Barrier(comm); CheckError(info);

	/* Set c_y */
	PLA_Copy_vector(rho, c_y);

	/* Compute corrector step */
	parallel_solve_reduced(h_x, h_y, a, c_x, c_y, &delta_x, &delta_y, &y1t, 
			       CORRECTOR);
	
	info = MPI_Barrier(comm); CheckError(info);
	/* Backsubstitution. */
	for(i=0; i<local_n; ++i)
	  {
	    idx = i*local_stride;
	    loc.delta_s[idx] = loc.s[idx] * 
	      (loc.delta_x[idx] - loc.hat_tau[idx]) / loc.t[idx];
	    loc.delta_z[idx] = loc.z[idx] *
	      (loc.hat_nu[idx] - loc.delta_x[idx]) / loc.g[idx];

	    loc.delta_g[idx] = loc.g[idx] *
	      (loc.gamma_z[idx] - loc.delta_z[idx]) / loc.z[idx];
	    loc.delta_t[idx] = loc.t[idx] *
	      (loc.gamma_s[idx] - loc.delta_s[idx]) / loc.s[idx];

	  }

	/* Update alfa. */
	for(i=0; i<local_n; ++i)
	  {
	    idx = i*local_stride;
	    loc.max_ratio[idx] = -loc.delta_g[idx] / loc.g[idx];
	    next_ratio = -loc.delta_t[idx] / loc.t[idx];
	    if(loc.max_ratio[idx] < next_ratio)
	      loc.max_ratio[idx] = next_ratio;
	    next_ratio = -loc.delta_s[idx] / loc.s[idx];
	    if(loc.max_ratio[idx] < next_ratio)
	      loc.max_ratio[idx] = next_ratio;
	    next_ratio = -loc.delta_z[idx] / loc.z[idx];
	    if(loc.max_ratio[idx] < next_ratio)
	      loc.max_ratio[idx] = next_ratio;
	  }
	info = MPI_Barrier(comm); CheckError(info);

	/* Note: There is a bug in the plapack documentation!
	 * Correct calling sequence is:
	 * int PLA_Iamax( PLA_Obj x, PLA_Obj k, PLA_Obj xmax). */
	info = PLA_Iamax(max_ratio, max_idx, alfa); CheckError(info);
	info = MPI_Barrier(comm); CheckError(info);
	*loc.alfa = -max(*loc.alfa, 1);
	*loc.alfa = (margin - 1) / (*loc.alfa);
	
	/* Set mu <- (s^T * t + z^T * g) / (2*n) * (
	   (alfa-1) / (alfa + 10))^2. */
	info = PLA_Dot(*z, *g, mu); CheckError(info);
	info = PLA_Dot(*s, *t, tmp); CheckError(info);
	info = MPI_Barrier(comm); CheckError(info);
	*loc.mu = (*loc.mu + (*loc.tmp)) / (2*n);
	*loc.mu = *loc.mu * sqr((*loc.alfa - 1) / (*loc.alfa + 10));

	/* Update variables. */
	info = PLA_Axpy(alfa, delta_x, *x); CheckError(info);
	info = PLA_Axpy(alfa, delta_g, *g); CheckError(info);
	info = PLA_Axpy(alfa, delta_t, *t); CheckError(info);
	info = PLA_Axpy(alfa, delta_z, *z); CheckError(info);
	info = PLA_Axpy(alfa, delta_s, *s); CheckError(info);
	info = PLA_Axpy(alfa, delta_y, *y); CheckError(info);
      } /* if(status == 0) */

    } /* while(status == STILL_RUNNING) */

  if(rank == 0 && status == 1 && verb >= STATUS)
    {
      printf("-----------------------------------------");
      printf("-----------------------------------------\n");
      printf("optimization converged\n");
    }

  info = MPI_Barrier(comm); CheckError(info);
  /* Compute dist <- c + h_dot_x - A^T*y */
  PLA_Copy_vector(h_dot_x, *dist);
  info = PLA_Gemv(PLA_TRANS, minus_one, a, *y, plus_one, *dist);
  CheckError(info);
  info = PLA_Axpy(plus_one, c, *dist); CheckError(info);

  info = MPI_Barrier(comm); CheckError(info);
  /* Deallocation */
  info = PLA_Obj_free(&diag_h_x); CheckError(info);  
  info = PLA_Obj_free(&h_y); CheckError(info);  
  info = PLA_Obj_free(&y1t); CheckError(info);  
  info = PLA_Obj_free(&c_x); CheckError(info);
  info = PLA_Obj_free(&c_y); CheckError(info);
  info = PLA_Obj_free(&h_dot_x); CheckError(info);
  info = PLA_Obj_free(&rho); CheckError(info);
  info = PLA_Obj_free(&nu); CheckError(info);
  info = PLA_Obj_free(&tau); CheckError(info);
  info = PLA_Obj_free(&sigma); CheckError(info);
  info = PLA_Obj_free(&gamma_z); CheckError(info);
  info = PLA_Obj_free(&gamma_s); CheckError(info);
  info = PLA_Obj_free(&hat_nu); CheckError(info);
  info = PLA_Obj_free(&hat_tau); CheckError(info);
  info = PLA_Obj_free(&delta_x); CheckError(info);
  info = PLA_Obj_free(&delta_y); CheckError(info);
  info = PLA_Obj_free(&delta_s); CheckError(info);
  info = PLA_Obj_free(&delta_z); CheckError(info);
  info = PLA_Obj_free(&delta_g); CheckError(info);
  info = PLA_Obj_free(&delta_t); CheckError(info);
  info = PLA_Obj_free(&d); CheckError(info);
  info = PLA_Obj_free(&ones_n); CheckError(info);
  info = PLA_Obj_free(&ones_m); CheckError(info);
  info = PLA_Obj_free(&h_x_copy); CheckError(info);
  info = PLA_Obj_free(&plus_one); CheckError(info);
  info = PLA_Obj_free(&minus_one); CheckError(info);
  info = PLA_Obj_free(&b_plus_1); CheckError(info);
  info = PLA_Obj_free(&c_plus_1); CheckError(info);
  info = PLA_Obj_free(&x_h_x); CheckError(info);
  info = PLA_Obj_free(&primal_inf); CheckError(info);
  info = PLA_Obj_free(&dual_inf); CheckError(info);
  info = PLA_Obj_free(&primal_obj); CheckError(info);
  info = PLA_Obj_free(&dual_obj); CheckError(info);
  info = PLA_Obj_free(&mu); CheckError(info);
  info = PLA_Obj_free(&alfa); CheckError(info);
  info = PLA_Obj_free(&tmp); CheckError(info);

  return status;
}
