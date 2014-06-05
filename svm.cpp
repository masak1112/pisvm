
//#include "psmo_solver.h"
//#include "loqo_solver.h"
#include "util.h"

#include <stdarg.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <mpi.h>
#include "svm.h"
#include "svm_cache.h"
#include "svm_kernel.h"
#include "svm_solver.h"
#include "svm_q_kernels.h"
#include "svm_solver_nu.h" //Needed for psmo_solver


#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

inline double clocks2sec(clock_t t)
{
    return (double) t / CLOCKS_PER_SEC;
}
#if 1
void info(const char *fmt,...)
{
    va_list ap;
    va_start(ap,fmt);
    vprintf(fmt,ap);
    va_end(ap);
}
void info_flush()
{
    fflush(stdout);
}
#else
void info(char *fmt,...) {}
void info_flush() {}
#endif



class SVQ_No_Cache : public QMatrix
{
public:
    SVQ_No_Cache(Qfloat *Q_bb_, Qfloat *QD_, const int n_);
    Qfloat *get_Q(int i, int len) const;
    Qfloat *get_Q_subset(int i, int *idxs, int n) const;
    Qfloat *get_QD() const;
    Qfloat get_non_cached(int i, int j) const;
    inline bool is_cached(const int i) const;
    void swap_index(int i, int j) const;
    virtual ~SVQ_No_Cache();
private:
    Qfloat *Q_bb;
    Qfloat *QD;
    int *idx;
    int n;
};

SVQ_No_Cache::SVQ_No_Cache(Qfloat *Q_bb_, Qfloat *QD_, const int n_)
{
    this->Q_bb = Q_bb_;
    this->QD = QD_;
    this->n = n_;
    this->idx = new int[n];
    for(int i=0; i<n; ++i)
        idx[i] = i;
}

Qfloat *SVQ_No_Cache::get_Q(int i, int len) const
{
    return &Q_bb[n*idx[i]];
}

Qfloat *SVQ_No_Cache::get_Q_subset(int i, int *idxs, int n) const
{
    printf("Error: Not implemented yet!\n");
    exit(1);
    return NULL;
}

Qfloat *SVQ_No_Cache::get_QD() const
{
    return QD;
}

Qfloat SVQ_No_Cache::get_non_cached(int i, int j) const
{
    return Q_bb[i*n+j];
}
inline bool SVQ_No_Cache::is_cached(const int i) const
{
    return false;
}

void SVQ_No_Cache::swap_index(int i, int j) const
{
    swap(idx[i], idx[j]);
    swap(QD[i], QD[j]);
}

SVQ_No_Cache::~SVQ_No_Cache()
{
    delete[] idx;
}

#ifdef SOLVER_PSMO
#include "psmo/psmo_solver.cpp"
#endif
//#include "gpm/gpm_solver.cpp"
#ifdef SOLVER_LOQO
#include "loqo/loqo_solver.cpp"
#endif



//
// construct and solve various formulations
//
static void solve_c_svc(const svm_problem *prob, const svm_parameter* param,
                        double *alpha, Solver::SolutionInfo* si,
                        double Cp, double Cn)
{
    int l = prob->l;
    double *minus_ones = new double[l];
    schar *y = new schar[l];

    int i;

    for(i=0; i<l; i++)
    {
        alpha[i] = 0;
        minus_ones[i] = -1;
        if(prob->y[i] > 0) y[i] = +1;
        else y[i]=-1;
    }

//   Solver_GPM sgpm(param->o, param->q);
//   sgpm.Solve(l, SVC_Q(*prob,*param,y), minus_ones, y,
// 	     alpha, Cp, Cn, param->eps, si, param->shrinking);
//   Solver_LOQO sl(param->o, param->q, 1);
//   sl.Solve(l, SVC_Q(*prob,*param,y), minus_ones, y,
//  	   alpha, Cp, Cn, param->eps, si, param->shrinking);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

#ifdef SOLVER_PSMO
    Solver_Parallel_SMO sps(param->o, param->q, MPI_COMM_WORLD);
    sps.Solve(l, SVC_Q(*prob,*param,y), minus_ones, y,
              alpha, Cp, Cn, param->eps, si, param->shrinking);
#endif
#ifdef SOLVER_LOQO
    Solver_Parallel_LOQO spl(param->o, param->q, 1, MPI_COMM_WORLD,
                             size, 1, param->o/size);
    spl.Solve(l, SVC_Q(*prob,*param,y), minus_ones, y,
              alpha, Cp, Cn, param->eps, si, param->shrinking);
#endif

//    Solver s;
//    s.Solve(l, SVC_Q(*prob,*param,y), minus_ones, y,
// 	   alpha, Cp, Cn, param->eps, si, param->shrinking);

    double sum_alpha=0;
    for(i=0; i<l; i++)
        sum_alpha += alpha[i];

    if (Cp==Cn)
        info("nu = %f\n", sum_alpha/(Cp*prob->l));

    for(i=0; i<l; i++)
        alpha[i] *= y[i];

    delete[] minus_ones;
    delete[] y;
}

static void solve_nu_svc(const svm_problem *prob, const svm_parameter *param,
                         double *alpha, Solver::SolutionInfo* si)
{
    int i;
    int l = prob->l;
    double nu = param->nu;

    schar *y = new schar[l];

    for(i=0; i<l; i++)
        if(prob->y[i]>0)
        {
            y[i] = +1;
        }
        else
        {
            y[i] = -1;
        }

    double sum_pos = nu*l/2;
    double sum_neg = nu*l/2;

    for(i=0; i<l; i++)
        if(y[i] == +1)
        {
            alpha[i] = min(1.0,sum_pos);
            sum_pos -= alpha[i];
        }
        else
        {
            alpha[i] = min(1.0,sum_neg);
            sum_neg -= alpha[i];
        }
    double *zeros = new double[l];

    for(i=0; i<l; i++)
    {
        zeros[i] = 0;
    }

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

#ifdef SOLVER_PSMO
    Solver_Parallel_SMO_NU sps(param->o, param->q, MPI_COMM_WORLD);
    sps.Solve(l, SVC_Q(*prob,*param,y), zeros, y,
              alpha, 1.0, 1.0, param->eps, si,  param->shrinking);
#endif
#ifdef SOLVER_LOQO
    Solver_Parallel_LOQO_NU spl(param->o, param->q, 2, MPI_COMM_WORLD,
                                size, 1, param->o/size, nu);
    spl.Solve(l, SVC_Q(*prob,*param,y), zeros, y,
              alpha, 1.0, 1.0, param->eps, si,  param->shrinking);
#endif

    // Serial solvers:
    //   Solver_LOQO_NU sl(param->o, param->q, 2, param->nu);
    //   sl.Solve(l, SVC_Q(*prob,*param,y), zeros, y,
    //  	     alpha, 1.0, 1.0, param->eps, si,  param->shrinking);
    //   Solver_NU s;
    //   s.Solve(l, SVC_Q(*prob,*param,y), zeros, y,
    // 	  alpha, 1.0, 1.0, param->eps, si,  param->shrinking);
    //
    //   Solver_GPM_NU sgpm(param->o, param->q);
    //   sgpm.Solve(l, SVC_Q_LOQO(*prob,*param,y), zeros, y,
    // 	     alpha, 1.0, 1.0, param->eps, si,  param->shrinking);

    double r = si->r;
    printf("si->r %g\n",si->r);
    info("C = %f\n",1/r);

//   printf("alphan ");
    for(i=0; i<l; i++)
    {
        alpha[i] *= y[i]/r;
//       printf(" %g", alpha[i]);
    }
//   printf("\n");


    si->rho /= r;
    si->obj /= (r*r);
    si->upper_bound_p = 1/r;
    si->upper_bound_n = 1/r;

    printf("si->rho = %g", si->rho);
    printf("si->r = %g", si->r);

    delete[] y;
    delete[] zeros;
}

static void solve_one_class(const svm_problem *prob, const svm_parameter *param,
                            double *alpha, Solver::SolutionInfo* si)
{
    int l = prob->l;
    double *zeros = new double[l];
    schar *ones = new schar[l];
    int i;

    int n = (int)(param->nu*prob->l);	// # of alpha's at upper bound

    for(i=0; i<n; i++)
        alpha[i] = 1;
    if(n<prob->l)
        alpha[n] = param->nu * prob->l - n;
    for(i=n+1; i<l; i++)
        alpha[i] = 0;

    for(i=0; i<l; i++)
    {
        zeros[i] = 0;
        ones[i] = 1;
    }

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Note: The following has not been tested but
    // should work.
#ifdef SOLVER_PSMO
    Solver_Parallel_SMO sps(param->o, param->q, MPI_COMM_WORLD);
    sps.Solve(l, ONE_CLASS_Q(*prob,*param), zeros, ones,
              alpha, 1.0, 1.0, param->eps, si,  param->shrinking);
#endif
#ifdef SOLVER_LOQO
    Solver_Parallel_LOQO spl(param->o, param->q, 2, MPI_COMM_WORLD,
                             size, 1, param->o/size);
    spl.Solve(l, ONE_CLASS_Q(*prob,*param), zeros, ones,
              alpha, 1.0, 1.0, param->eps, si,  param->shrinking);
#endif

    // Serial solver:
    //   Solver s;
    //   s.Solve(l, ONE_CLASS_Q(*prob,*param), zeros, ones,
    //  	  alpha, 1.0, 1.0, param->eps, si, param->shrinking);

    delete[] zeros;
    delete[] ones;
}

static void solve_epsilon_svr(const svm_problem *prob, const svm_parameter *param,
                              double *alpha, Solver::SolutionInfo* si)
{
    int l = prob->l;
    double *alpha2 = new double[2*l];
    double *linear_term = new double[2*l];
    schar *y = new schar[2*l];
    int i;

    for(i=0; i<l; i++)
    {
        alpha2[i] = 0;
        linear_term[i] = param->p - prob->y[i];
        y[i] = 1;

        alpha2[i+l] = 0;
        linear_term[i+l] = param->p + prob->y[i];
        y[i+l] = -1;
    }

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

#ifdef SOLVER_PSMO
    Solver_Parallel_SMO sps(param->o, param->q, MPI_COMM_WORLD);
    sps.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
              alpha2, param->C, param->C, param->eps, si, param->shrinking);
#endif
#ifdef SOLVER_LOQO
    Solver_Parallel_LOQO spl(param->o, param->q, 1, MPI_COMM_WORLD,
                             size, 1, param->o/size);
    spl.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
              alpha2, param->C, param->C, param->eps, si, param->shrinking);
#endif

    // Serial solver
    //   Solver s;
    //   s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
    // 	  alpha2, param->C, param->C, param->eps, si, param->shrinking);


    double sum_alpha = 0;
    for(i=0; i<l; i++)
    {
        alpha[i] = alpha2[i] - alpha2[i+l];
        sum_alpha += fabs(alpha[i]);
    }
    info("nu = %f\n",sum_alpha/(param->C*l));

    delete[] alpha2;
    delete[] linear_term;
    delete[] y;
}

static void solve_nu_svr(const svm_problem *prob, const svm_parameter *param,
                         double *alpha, Solver::SolutionInfo* si)
{
    int l = prob->l;
    double C = param->C;
    double *alpha2 = new double[2*l];
    double *linear_term = new double[2*l];
    schar *y = new schar[2*l];
    int i;

    double sum = C * param->nu * l / 2;
    for(i=0; i<l; i++)
    {
        alpha2[i] = alpha2[i+l] = min(sum,C);
        sum -= alpha2[i];

        linear_term[i] = - prob->y[i];
        y[i] = 1;

        linear_term[i+l] = prob->y[i];
        y[i+l] = -1;
    }
//   printf("alpha = ");
//   for(i = 0; i<2*l; ++i)
//     printf(" %g", alpha2[i]);
//   printf("\n");

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

#ifdef SOLVER_PSMO
    Solver_Parallel_SMO_NU sps(param->o, param->q, MPI_COMM_WORLD);
    sps.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
              alpha2, C, C, param->eps, si, param->shrinking);
#endif
#ifdef SOLVER_LOQO
    Solver_Parallel_LOQO_NU spl(param->o, param->q, 2, MPI_COMM_WORLD,
                                size, 1, param->o/size, C*param->nu/2);
    spl.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
              alpha2, C, C, param->eps, si,  param->shrinking);
#endif

    // Serial solvers:
    //   Solver_LOQO_NU sl(param->o, param->q, 2, C*param->nu/2);
    //   sl.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
    // 	   alpha2, C, C, param->eps, si,  param->shrinking);
    //   Solver_NU s;
    //   s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
    // 	  alpha2, C, C, param->eps, si, param->shrinking);

    info("epsilon = %f\n",-si->r);

    for(i=0; i<l; i++)
        alpha[i] = alpha2[i] - alpha2[i+l];

    delete[] alpha2;
    delete[] linear_term;
    delete[] y;
}

//
// decision_function
//
struct decision_function
{
    double *alpha;
    double rho;
};

decision_function svm_train_one(const svm_problem *prob,
                                const svm_parameter *param,
                                double Cp, double Cn)
{
    double *alpha = Malloc(double,prob->l);
    Solver::SolutionInfo si;
    switch(param->svm_type)
    {
    case C_SVC:
        solve_c_svc(prob,param,alpha,&si,Cp,Cn);
        break;
    case NU_SVC:
        solve_nu_svc(prob,param,alpha,&si);
        break;
    case ONE_CLASS:
        solve_one_class(prob,param,alpha,&si);
        break;
    case EPSILON_SVR:
        solve_epsilon_svr(prob,param,alpha,&si);
        break;
    case NU_SVR:
        solve_nu_svr(prob,param,alpha,&si);
        break;
    }

    info("obj = %f, rho = %f\n",si.obj,si.rho);

    // output SVs

    int nSV = 0;
    int nBSV = 0;
    for(int i=0; i<prob->l; i++)
    {
        if(fabs(alpha[i]) > 0)
        {
            ++nSV;
            if(prob->y[i] > 0)
            {
                if(fabs(alpha[i]) >= si.upper_bound_p)
                    ++nBSV;
            }
            else
            {
                if(fabs(alpha[i]) >= si.upper_bound_n)
                    ++nBSV;
            }
        }
    }

    info("nSV = %d, nBSV = %d\n",nSV,nBSV);

    decision_function f;
    f.alpha = alpha;
    f.rho = si.rho;
    return f;
}

//
// svm_model
//
struct svm_model
{
    svm_parameter param;	// parameter
    int nr_class;		// number of classes, = 2 in regression/one class svm
    int l;			// total #SV
    Xfloat **SV;
    int **nz_sv;
    int *sv_len;
    int max_idx;
    double **sv_coef;	// coefficients for SVs in decision functions (sv_coef[n-1][l])
    double *rho;		// constants in decision functions (rho[n*(n-1)/2])
    double *probA;          // pariwise probability information
    double *probB;

    // for classification only

    int *label;		// label of each class (label[n])
    int *nSV;		// number of SVs for each class (nSV[n])
    // nSV[0] + nSV[1] + ... + nSV[n-1] = l
    // XXX
    int free_sv;		// 1 if svm_model is created by svm_load_model
    // 0 if svm_model is created by svm_train
};

// Platt's binary SVM Probablistic Output: an improvement from Lin et al.
void sigmoid_train(int l, const double *dec_values, const double *labels,
                   double& A, double& B)
{
    double prior1=0, prior0 = 0;
    int i;

    for (i=0; i<l; i++)
        if (labels[i] > 0) prior1+=1;
        else prior0+=1;

    int max_iter=100; 	// Maximal number of iterations
    double min_step=1e-10;	// Minimal step taken in line search
    double sigma=1e-3;	// For numerically strict PD of Hessian
    double eps=1e-5;
    double hiTarget=(prior1+1.0)/(prior1+2.0);
    double loTarget=1/(prior0+2.0);
    double *t=Malloc(double,l);
    double fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
    double newA,newB,newf,d1,d2;
    int iter;

    // Initial Point and Initial Fun Value
    A=0.0;
    B=log((prior0+1.0)/(prior1+1.0));
    double fval = 0.0;

    for (i=0; i<l; i++)
    {
        if (labels[i]>0) t[i]=hiTarget;
        else t[i]=loTarget;
        fApB = dec_values[i]*A+B;
        if (fApB>=0)
            fval += t[i]*fApB + log(1+exp(-fApB));
        else
            fval += (t[i] - 1)*fApB +log(1+exp(fApB));
    }
    for (iter=0; iter<max_iter; iter++)
    {
        // Update Gradient and Hessian (use H' = H + sigma I)
        h11=sigma; // numerically ensures strict PD
        h22=sigma;
        h21=0.0;
        g1=0.0;
        g2=0.0;
        for (i=0; i<l; i++)
        {
            fApB = dec_values[i]*A+B;
            if (fApB >= 0)
            {
                p=exp(-fApB)/(1.0+exp(-fApB));
                q=1.0/(1.0+exp(-fApB));
            }
            else
            {
                p=1.0/(1.0+exp(fApB));
                q=exp(fApB)/(1.0+exp(fApB));
            }
            d2=p*q;
            h11+=dec_values[i]*dec_values[i]*d2;
            h22+=d2;
            h21+=dec_values[i]*d2;
            d1=t[i]-p;
            g1+=dec_values[i]*d1;
            g2+=d1;
        }

        // Stopping Criteria
        if (fabs(g1)<eps && fabs(g2)<eps)
            break;

        // Finding Newton direction: -inv(H') * g
        det=h11*h22-h21*h21;
        dA=-(h22*g1 - h21 * g2) / det;
        dB=-(-h21*g1+ h11 * g2) / det;
        gd=g1*dA+g2*dB;


        stepsize = 1; 		// Line Search
        while (stepsize >= min_step)
        {
            newA = A + stepsize * dA;
            newB = B + stepsize * dB;

            // New function value
            newf = 0.0;
            for (i=0; i<l; i++)
            {
                fApB = dec_values[i]*newA+newB;
                if (fApB >= 0)
                    newf += t[i]*fApB + log(1+exp(-fApB));
                else
                    newf += (t[i] - 1)*fApB +log(1+exp(fApB));
            }
            // Check sufficient decrease
            if (newf<fval+0.0001*stepsize*gd)
            {
                A=newA;
                B=newB;
                fval=newf;
                break;
            }
            else
                stepsize = stepsize / 2.0;
        }

        if (stepsize < min_step)
        {
            info("Line search fails in two-class probability estimates\n");
            break;
        }
    }

    if (iter>=max_iter)
        info("Reaching maximal iterations in two-class probability estimates\n");
    free(t);
}

double sigmoid_predict(double decision_value, double A, double B)
{
    double fApB = decision_value*A+B;
    if (fApB >= 0)
        return exp(-fApB)/(1.0+exp(-fApB));
    else
        return 1.0/(1+exp(fApB)) ;
}

// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
void multiclass_probability(int k, double **r, double *p)
{
    int t,j;
    int iter = 0, max_iter=100;
    double **Q=Malloc(double *,k);
    double *Qp=Malloc(double,k);
    double pQp, eps=0.005/k;

    for (t=0; t<k; t++)
    {
        p[t]=1.0/k;  // Valid if k = 1
        Q[t]=Malloc(double,k);
        Q[t][t]=0;
        for (j=0; j<t; j++)
        {
            Q[t][t]+=r[j][t]*r[j][t];
            Q[t][j]=Q[j][t];
        }
        for (j=t+1; j<k; j++)
        {
            Q[t][t]+=r[j][t]*r[j][t];
            Q[t][j]=-r[j][t]*r[t][j];
        }
    }
    for (iter=0; iter<max_iter; iter++)
    {
        // stopping condition, recalculate QP,pQP for numerical accuracy
        pQp=0;
        for (t=0; t<k; t++)
        {
            Qp[t]=0;
            for (j=0; j<k; j++)
                Qp[t]+=Q[t][j]*p[j];
            pQp+=p[t]*Qp[t];
        }
        double max_error=0;
        for (t=0; t<k; t++)
        {
            double error=fabs(Qp[t]-pQp);
            if (error>max_error)
                max_error=error;
        }
        if (max_error<eps) break;

        for (t=0; t<k; t++)
        {
            double diff=(-Qp[t]+pQp)/Q[t][t];
            p[t]+=diff;
            pQp=(pQp+diff*(diff*Q[t][t]+2*Qp[t]))/(1+diff)/(1+diff);
            for (j=0; j<k; j++)
            {
                Qp[j]=(Qp[j]+diff*Q[t][j])/(1+diff);
                p[j]/=(1+diff);
            }
        }
    }
    if (iter>=max_iter)
        info("Exceeds max_iter in multiclass_prob\n");
    for(t=0; t<k; t++) free(Q[t]);
    free(Q);
    free(Qp);
}

// Cross-validation decision values for probability estimates
void svm_binary_svc_probability(const svm_problem *prob,
                                const svm_parameter *param,
                                double Cp, double Cn,
                                double& probA, double& probB)
{
    int i;
    int nr_fold = 5;
    int *perm = Malloc(int,prob->l);
    double *dec_values = Malloc(double,prob->l);

    // random shuffle
    for(i=0; i<prob->l; i++) perm[i]=i;
    for(i=0; i<prob->l; i++)
    {
        int j = i+rand()%(prob->l-i);
        swap(perm[i],perm[j]);
    }
    for(i=0; i<nr_fold; i++)
    {
        int begin = i*prob->l/nr_fold;
        int end = (i+1)*prob->l/nr_fold;
        int j,k;
        struct svm_problem subprob;

        subprob.l = prob->l-(end-begin);
        subprob.max_idx = prob->max_idx;
        subprob.x = Malloc(Xfloat *,subprob.l);
        subprob.nz_idx = Malloc(int *,subprob.l);
        subprob.x_len = Malloc(int,subprob.l);
        subprob.y = Malloc(double,subprob.l);

        k=0;
        for(j=0; j<begin; j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.nz_idx[k] = prob->nz_idx[perm[j]];
            subprob.x_len[k] = prob->x_len[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            ++k;
        }
        for(j=end; j<prob->l; j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.nz_idx[k] = prob->nz_idx[perm[j]];
            subprob.x_len[k] = prob->x_len[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            ++k;
        }
        int p_count=0,n_count=0;
        for(j=0; j<k; j++)
            if(subprob.y[j]>0)
                p_count++;
            else
                n_count++;

        if(p_count==0 && n_count==0)
            for(j=begin; j<end; j++)
                dec_values[perm[j]] = 0;
        else if(p_count > 0 && n_count == 0)
            for(j=begin; j<end; j++)
                dec_values[perm[j]] = 1;
        else if(p_count == 0 && n_count > 0)
            for(j=begin; j<end; j++)
                dec_values[perm[j]] = -1;
        else
        {
            svm_parameter subparam = *param;
            subparam.probability=0;
            subparam.C=1.0;
            subparam.nr_weight=2;
            subparam.weight_label = Malloc(int,2);
            subparam.weight = Malloc(double,2);
            subparam.weight_label[0]=+1;
            subparam.weight_label[1]=-1;
            subparam.weight[0]=Cp;
            subparam.weight[1]=Cn;
            struct svm_model *submodel = svm_train(&subprob,&subparam);
            for(j=begin; j<end; j++)
            {
                svm_predict_values(submodel,prob->x[perm[j]],
                                   prob->nz_idx[perm[j]], prob->x_len[perm[j]],
                                   &(dec_values[perm[j]]));
                // ensure +1 -1 order; reason not using CV subroutine
                dec_values[perm[j]] *= submodel->label[0];
            }
            svm_destroy_model(submodel);
            svm_destroy_param(&subparam);
            free(subprob.x);
            free(subprob.nz_idx);
            free(subprob.x_len);
            free(subprob.y);
        }
    }
    sigmoid_train(prob->l,dec_values,prob->y,probA,probB);
    free(dec_values);
    free(perm);
}

// Return parameter of a Laplace distribution
double svm_svr_probability(const svm_problem *prob, const svm_parameter *param)
{
    int i;
    int nr_fold = 5;
    double *ymv = Malloc(double,prob->l);
    double mae = 0;

    svm_parameter newparam = *param;
    newparam.probability = 0;
    svm_cross_validation(prob,&newparam,nr_fold,ymv);
    for(i=0; i<prob->l; i++)
    {
        ymv[i]=prob->y[i]-ymv[i];
        mae += fabs(ymv[i]);
    }
    mae /= prob->l;
    double std=sqrt(2*mae*mae);
    int count=0;
    mae=0;
    for(i=0; i<prob->l; i++)
        if (fabs(ymv[i]) > 5*std)
            count=count+1;
        else
            mae+=fabs(ymv[i]);
    mae /= (prob->l-count);
    info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= %g\n",mae);
    free(ymv);
    return mae;
}


// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
void svm_group_classes(const svm_problem *prob, int *nr_class_ret,
                       int **label_ret, int **start_ret, int **count_ret,
                       int *perm)
{
    int l = prob->l;
    int max_nr_class = 16;
    int nr_class = 0;
    int *label = Malloc(int,max_nr_class);
    int *count = Malloc(int,max_nr_class);
    int *data_label = Malloc(int,l);
    int i;

    for(i=0; i<l; i++)
    {
        int this_label = (int)prob->y[i];
        int j;
        for(j=0; j<nr_class; j++)
        {
            if(this_label == label[j])
            {
                ++count[j];
                break;
            }
        }
        data_label[i] = j;
        if(j == nr_class)
        {
            if(nr_class == max_nr_class)
            {
                max_nr_class *= 2;
                label = (int *)realloc(label,max_nr_class*sizeof(int));
                count = (int *)realloc(count,max_nr_class*sizeof(int));
            }
            label[nr_class] = this_label;
            count[nr_class] = 1;
            ++nr_class;
        }
    }

    int *start = Malloc(int,nr_class);
    start[0] = 0;
    for(i=1; i<nr_class; i++)
        start[i] = start[i-1]+count[i-1];
    for(i=0; i<l; i++)
    {
        perm[start[data_label[i]]] = i;
        ++start[data_label[i]];
    }
    start[0] = 0;
    for(i=1; i<nr_class; i++)
        start[i] = start[i-1]+count[i-1];

    *nr_class_ret = nr_class;
    *label_ret = label;
    *start_ret = start;
    *count_ret = count;
    free(data_label);
}

//
// Interface functions
//
svm_model *svm_train(const svm_problem *prob, const svm_parameter *param)
{
    svm_model *model = Malloc(svm_model,1);
    model->param = *param;
    model->free_sv = 0;	// XXX

    if(param->svm_type == ONE_CLASS ||
            param->svm_type == EPSILON_SVR ||
            param->svm_type == NU_SVR)
    {
        // regression or one-class-svm
        model->nr_class = 2;
        model->label = NULL;
        model->nSV = NULL;
        model->probA = NULL;
        model->probB = NULL;
        model->sv_coef = Malloc(double *,1);

        if(param->probability &&
                (param->svm_type == EPSILON_SVR ||
                 param->svm_type == NU_SVR))
        {
            model->probA = Malloc(double,1);
            model->probA[0] = svm_svr_probability(prob,param);
        }

        decision_function f = svm_train_one(prob,param,0,0);
        model->rho = Malloc(double,1);
        model->rho[0] = f.rho;

        int nSV = 0;
        int i;
        for(i=0; i<prob->l; i++)
            if(fabs(f.alpha[i]) > 0) ++nSV;
        model->l = nSV;
        model->SV = Malloc(Xfloat *,nSV);
        model->nz_sv = Malloc(int *,nSV);
        model->sv_len = Malloc(int,nSV);
        model->sv_coef[0] = Malloc(double,nSV);
        int j = 0;
        for(i=0; i<prob->l; i++)
            if(fabs(f.alpha[i]) > 0)
            {
                model->SV[j] = prob->x[i];
                model->nz_sv[j] = prob->nz_idx[i];
                model->sv_len[j] = prob->x_len[i];
                model->sv_coef[0][j] = f.alpha[i];
                ++j;
            }

        free(f.alpha);
    }
    else
    {
        // classification
        int l = prob->l;
        int nr_class;
        int *label = NULL;
        int *start = NULL;
        int *count = NULL;
        int *perm = Malloc(int,l);

        // group training data of the same class
        svm_group_classes(prob,&nr_class,&label,&start,&count,perm);

        Xfloat **x = Malloc(Xfloat *, l);
        int **nz_idx = Malloc(int *, l);
        int *x_len = Malloc(int, l);
        int i;
        for(i=0; i<l; i++)
        {
            x[i] = prob->x[perm[i]];
            nz_idx[i] = prob->nz_idx[perm[i]];
            x_len[i] = prob->x_len[perm[i]];
        }

        // calculate weighted C

        double *weighted_C = Malloc(double, nr_class);
        for(i=0; i<nr_class; i++)
            weighted_C[i] = param->C;
        for(i=0; i<param->nr_weight; i++)
        {
            int j;
            for(j=0; j<nr_class; j++)
                if(param->weight_label[i] == label[j])
                    break;
            if(j == nr_class)
                fprintf(stderr, "warning: class label %d specified in weight is not found\n", param->weight_label[i]);
            else
                weighted_C[j] *= param->weight[i];
        }

        // train k*(k-1)/2 models

        bool *nonzero = Malloc(bool,l);
        for(i=0; i<l; i++)
            nonzero[i] = false;
        decision_function *f = Malloc(decision_function,nr_class*(nr_class-1)/2);

        double *probA=NULL,*probB=NULL;
        if (param->probability)
        {
            probA=Malloc(double,nr_class*(nr_class-1)/2);
            probB=Malloc(double,nr_class*(nr_class-1)/2);
        }

        int p = 0;
        for(i=0; i<nr_class; i++)
            for(int j=i+1; j<nr_class; j++)
            {
                svm_problem sub_prob;
                int si = start[i], sj = start[j];
                int ci = count[i], cj = count[j];
                sub_prob.l = ci+cj;
                sub_prob.max_idx = prob->max_idx;
                sub_prob.x = Malloc(Xfloat *, sub_prob.l);
                sub_prob.nz_idx = Malloc(int *, sub_prob.l);
                sub_prob.x_len = Malloc(int, sub_prob.l);
                sub_prob.y = Malloc(double,sub_prob.l);
                int k;
                for(k=0; k<ci; k++)
                {
                    sub_prob.x[k] = x[si+k];
                    sub_prob.nz_idx[k] = nz_idx[si+k];
                    sub_prob.x_len[k] = x_len[si+k];
                    sub_prob.y[k] = +1;
                }
                for(k=0; k<cj; k++)
                {
                    sub_prob.x[ci+k] = x[sj+k];
                    sub_prob.nz_idx[ci+k] = nz_idx[sj+k];
                    sub_prob.x_len[ci+k] = x_len[sj+k];
                    sub_prob.y[ci+k] = -1;
                }

                if(param->probability)
                    svm_binary_svc_probability(&sub_prob,param,weighted_C[i],weighted_C[j],probA[p],probB[p]);

                f[p] = svm_train_one(&sub_prob,param,weighted_C[i],weighted_C[j]);
                for(k=0; k<ci; k++)
                    if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
                        nonzero[si+k] = true;
                for(k=0; k<cj; k++)
                    if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
                        nonzero[sj+k] = true;
                free(sub_prob.x);
                free(sub_prob.nz_idx);
                free(sub_prob.x_len);
                free(sub_prob.y);
                ++p;
            }

        // build output

        model->nr_class = nr_class;

        model->label = Malloc(int,nr_class);
        for(i=0; i<nr_class; i++)
            model->label[i] = label[i];

        model->rho = Malloc(double,nr_class*(nr_class-1)/2);
        for(i=0; i<nr_class*(nr_class-1)/2; i++)
            model->rho[i] = f[i].rho;

        if(param->probability)
        {
            model->probA = Malloc(double,nr_class*(nr_class-1)/2);
            model->probB = Malloc(double,nr_class*(nr_class-1)/2);
            for(i=0; i<nr_class*(nr_class-1)/2; i++)
            {
                model->probA[i] = probA[i];
                model->probB[i] = probB[i];
            }
        }
        else
        {
            model->probA=NULL;
            model->probB=NULL;
        }

        int total_sv = 0;
        int *nz_count = Malloc(int,nr_class);
        model->nSV = Malloc(int,nr_class);
        for(i=0; i<nr_class; i++)
        {
            int nSV = 0;
            for(int j=0; j<count[i]; j++)
                if(nonzero[start[i]+j])
                {
                    ++nSV;
                    ++total_sv;
                }
            model->nSV[i] = nSV;
            nz_count[i] = nSV;
        }

        info("Total nSV = %d\n",total_sv);

        model->l = total_sv;
        model->SV = Malloc(Xfloat *, total_sv);
        model->nz_sv = Malloc(int *, total_sv);
        model->sv_len = Malloc(int, total_sv);
        p = 0;
        for(i=0; i<l; i++)
            if(nonzero[i])
            {
                model->SV[p] = x[i];
                model->nz_sv[p] = nz_idx[i];
                model->sv_len[p] = x_len[i];
                ++p;
            }
        int *nz_start = Malloc(int,nr_class);
        nz_start[0] = 0;
        for(i=1; i<nr_class; i++)
            nz_start[i] = nz_start[i-1]+nz_count[i-1];

        model->sv_coef = Malloc(double *,nr_class-1);
        for(i=0; i<nr_class-1; i++)
            model->sv_coef[i] = Malloc(double,total_sv);

        p = 0;
        for(i=0; i<nr_class; i++)
            for(int j=i+1; j<nr_class; j++)
            {
                // classifier (i,j): coefficients with
                // i are in sv_coef[j-1][nz_start[i]...],
                // j are in sv_coef[i][nz_start[j]...]

                int si = start[i];
                int sj = start[j];
                int ci = count[i];
                int cj = count[j];

                int q = nz_start[i];
                int k;
                for(k=0; k<ci; k++)
                    if(nonzero[si+k])
                        model->sv_coef[j-1][q++] = f[p].alpha[k];
                q = nz_start[j];
                for(k=0; k<cj; k++)
                    if(nonzero[sj+k])
                        model->sv_coef[i][q++] = f[p].alpha[ci+k];
                ++p;
            }

        free(label);
        free(probA);
        free(probB);
        free(count);
        free(perm);
        free(start);
        free(x);
        free(nz_idx);
        free(x_len);
        free(weighted_C);
        free(nonzero);
        for(i=0; i<nr_class*(nr_class-1)/2; i++)
            free(f[i].alpha);
        free(f);
        free(nz_count);
        free(nz_start);
    }
    return model;
}

// Stratified cross validation
void svm_cross_validation(const svm_problem *prob, const svm_parameter *param,
                          int nr_fold, double *target)
{
    int i;
    int *fold_start = Malloc(int,nr_fold+1);
    int l = prob->l;
    int *perm = Malloc(int,l);
    int nr_class;

    // stratified cv may not give leave-one-out rate
    // Each class to l folds -> some folds may have zero elements
    if((param->svm_type == C_SVC ||
            param->svm_type == NU_SVC) && nr_fold < l)
    {
        int *start = NULL;
        int *label = NULL;
        int *count = NULL;
        svm_group_classes(prob,&nr_class,&label,&start,&count,perm);

        // random shuffle and then data grouped by fold using the array perm
        int *fold_count = Malloc(int,nr_fold);
        int c;
        int *index = Malloc(int,l);
        for(i=0; i<l; i++)
            index[i]=perm[i];
        for (c=0; c<nr_class; c++)
            for(i=0; i<count[c]; i++)
            {
                int j = i+rand()%(count[c]-i);
                swap(index[start[c]+j],index[start[c]+i]);
            }
        for(i=0; i<nr_fold; i++)
        {
            fold_count[i] = 0;
            for (c=0; c<nr_class; c++)
                fold_count[i]+=(i+1)*count[c]/nr_fold-i*count[c]/nr_fold;
        }
        fold_start[0]=0;
        for (i=1; i<=nr_fold; i++)
            fold_start[i] = fold_start[i-1]+fold_count[i-1];
        for (c=0; c<nr_class; c++)
            for(i=0; i<nr_fold; i++)
            {
                int begin = start[c]+i*count[c]/nr_fold;
                int end = start[c]+(i+1)*count[c]/nr_fold;
                for(int j=begin; j<end; j++)
                {
                    perm[fold_start[i]] = index[j];
                    fold_start[i]++;
                }
            }
        fold_start[0]=0;
        for (i=1; i<=nr_fold; i++)
            fold_start[i] = fold_start[i-1]+fold_count[i-1];
        free(start);
        free(label);
        free(count);
        free(index);
        free(fold_count);
    }
    else
    {
        for(i=0; i<l; i++) perm[i]=i;
        for(i=0; i<l; i++)
        {
            int j = i+rand()%(l-i);
            swap(perm[i],perm[j]);
        }
        for(i=0; i<=nr_fold; i++)
            fold_start[i]=i*l/nr_fold;
    }

    for(i=0; i<nr_fold; i++)
    {
        int begin = fold_start[i];
        int end = fold_start[i+1];
        int j,k;
        struct svm_problem subprob;

        subprob.l = l-(end-begin);
        subprob.x = Malloc(Xfloat *, subprob.l);
        subprob.nz_idx = Malloc(int *, subprob.l);
        subprob.x_len = Malloc(int, subprob.l);
        subprob.y = Malloc(double,subprob.l);

        k=0;
        for(j=0; j<begin; j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.nz_idx[k] = prob->nz_idx[perm[j]];
            subprob.x_len[k] = prob->x_len[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            ++k;
        }
        for(j=end; j<l; j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.nz_idx[k] = prob->nz_idx[perm[j]];
            subprob.x_len[k] = prob->x_len[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            ++k;
        }
        struct svm_model *submodel = svm_train(&subprob,param);
        if(param->probability &&
                (param->svm_type == C_SVC || param->svm_type == NU_SVC))
        {
            double *prob_estimates=Malloc(double,svm_get_nr_class(submodel));
            for(j=begin; j<end; j++)
                target[perm[j]] =
                    svm_predict_probability(submodel,prob->x[perm[j]],
                                            prob->nz_idx[perm[j]],
                                            prob->x_len[perm[j]],
                                            prob_estimates);
            free(prob_estimates);
        }
        else
            for(j=begin; j<end; j++)
                target[perm[j]] = svm_predict(submodel,prob->x[perm[j]],
                                              prob->nz_idx[perm[j]],
                                              prob->x_len[perm[j]]);
        svm_destroy_model(submodel);
        free(subprob.x);
        free(subprob.nz_idx);
        free(subprob.x_len);
        free(subprob.y);
    }
    free(fold_start);
    free(perm);
}


int svm_get_svm_type(const svm_model *model)
{
    return model->param.svm_type;
}

int svm_get_nr_class(const svm_model *model)
{
    return model->nr_class;
}

void svm_get_labels(const svm_model *model, int* label)
{
    if (model->label != NULL)
        for(int i=0; i<model->nr_class; i++)
            label[i] = model->label[i];
}

double svm_get_svr_probability(const svm_model *model)
{
    if ((model->param.svm_type == EPSILON_SVR ||
            model->param.svm_type == NU_SVR) &&
            model->probA!=NULL)
        return model->probA[0];
    else
    {
        info("Model doesn't contain information for SVR probability inference\n");
        return 0;
    }
}

void svm_predict_values(const svm_model *model, const Xfloat *x,
                        const int *nz_x, const int lx, double* dec_values)
{
    if(model->param.svm_type == ONE_CLASS ||
            model->param.svm_type == EPSILON_SVR ||
            model->param.svm_type == NU_SVR)
    {
        double *sv_coef = model->sv_coef[0];
        double sum = 0;
        for(int i=0; i<model->l; i++)
            sum += sv_coef[i] *
                   Kernel::k_function(x, nz_x, lx,
                                      model->SV[i], model->nz_sv[i], model->sv_len[i],
                                      model->param);
        sum -= model->rho[0];
        *dec_values = sum;
    }
    else
    {
        int i;
        int nr_class = model->nr_class;
        int l = model->l;

        double *kvalue = Malloc(double,l);
        for(i=0; i<l; i++)
            kvalue[i] = Kernel::k_function(x, nz_x, lx,
                                           model->SV[i], model->nz_sv[i],
                                           model->sv_len[i], model->param);

        int *start = Malloc(int,nr_class);
        start[0] = 0;
        for(i=1; i<nr_class; i++)
            start[i] = start[i-1]+model->nSV[i-1];

        int p=0;
        int pos=0;
        for(i=0; i<nr_class; i++)
            for(int j=i+1; j<nr_class; j++)
            {
                double sum = 0;
                int si = start[i];
                int sj = start[j];
                int ci = model->nSV[i];
                int cj = model->nSV[j];

                int k;
                double *coef1 = model->sv_coef[j-1];
                double *coef2 = model->sv_coef[i];
                for(k=0; k<ci; k++)
                    sum += coef1[si+k] * kvalue[si+k];
                for(k=0; k<cj; k++)
                    sum += coef2[sj+k] * kvalue[sj+k];
                sum -= model->rho[p++];
                dec_values[pos++] = sum;
            }

        free(kvalue);
        free(start);
    }
}

double svm_predict(const svm_model *model, const Xfloat *x, const int *nz_x,
                   const int lx)
{
    if(model->param.svm_type == ONE_CLASS ||
            model->param.svm_type == EPSILON_SVR ||
            model->param.svm_type == NU_SVR)
    {
        double res;
        svm_predict_values(model, x, nz_x, lx, &res);

        if(model->param.svm_type == ONE_CLASS)
            return (res>0)?1:-1;
        else
            return res;
    }
    else
    {
        int i;
        int nr_class = model->nr_class;
        double *dec_values = Malloc(double, nr_class*(nr_class-1)/2);
        svm_predict_values(model, x, nz_x, lx, dec_values);

        int *vote = Malloc(int,nr_class);
        for(i=0; i<nr_class; i++)
            vote[i] = 0;
        int pos=0;
        for(i=0; i<nr_class; i++)
            for(int j=i+1; j<nr_class; j++)
            {
                if(dec_values[pos++] > 0)
                    ++vote[i];
                else
                    ++vote[j];
            }

        int vote_max_idx = 0;
        for(i=1; i<nr_class; i++)
            if(vote[i] > vote[vote_max_idx])
                vote_max_idx = i;
        free(vote);
        free(dec_values);
        return model->label[vote_max_idx];
    }
}

double svm_predict_probability(const svm_model *model, const Xfloat *x,
                               const int *nz_x, const int lx,
                               double *prob_estimates)
{
    if ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
            model->probA!=NULL && model->probB!=NULL)
    {
        int i;
        int nr_class = model->nr_class;
        double *dec_values = Malloc(double, nr_class*(nr_class-1)/2);
        svm_predict_values(model, x, nz_x, lx, dec_values);

        double min_prob=1e-7;
        double **pairwise_prob=Malloc(double *,nr_class);
        for(i=0; i<nr_class; i++)
            pairwise_prob[i]=Malloc(double,nr_class);
        int k=0;
        for(i=0; i<nr_class; i++)
            for(int j=i+1; j<nr_class; j++)
            {
                pairwise_prob[i][j]=
                    min(max(sigmoid_predict(dec_values[k],model->probA[k],
                                            model->probB[k]),min_prob),1-min_prob);
                pairwise_prob[j][i]=1-pairwise_prob[i][j];
                k++;
            }
        multiclass_probability(nr_class,pairwise_prob,prob_estimates);

        int prob_max_idx = 0;
        for(i=1; i<nr_class; i++)
            if(prob_estimates[i] > prob_estimates[prob_max_idx])
                prob_max_idx = i;
        for(i=0; i<nr_class; i++)
            free(pairwise_prob[i]);
        free(dec_values);
        free(pairwise_prob);
        return model->label[prob_max_idx];
    }
    else
        return svm_predict(model, x, nz_x, lx);
}

const char *svm_type_table[] =
{
    "c_svc","nu_svc","one_class","epsilon_svr","nu_svr",NULL
};

const char *kernel_type_table[]=
{
    "linear","polynomial","rbf","sigmoid",NULL
};

int svm_save_model(const char *model_file_name, const svm_model *model)
{
    FILE *fp = fopen(model_file_name,"w");
    if(fp==NULL) return -1;

    const svm_parameter& param = model->param;

    fprintf(fp,"svm_type %s\n", svm_type_table[param.svm_type]);
    fprintf(fp,"kernel_type %s\n", kernel_type_table[param.kernel_type]);

    if(param.kernel_type == POLY)
        fprintf(fp,"degree %d\n", param.degree);

    if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
        fprintf(fp,"gamma %g\n", param.gamma);

    if(param.kernel_type == POLY || param.kernel_type == SIGMOID)
        fprintf(fp,"coef0 %g\n", param.coef0);

    int nr_class = model->nr_class;
    int l = model->l;
    fprintf(fp, "nr_class %d\n", nr_class);
    fprintf(fp, "total_sv %d\n",l);

    {
        fprintf(fp, "rho");
        for(int i=0; i<nr_class*(nr_class-1)/2; i++)
            fprintf(fp," %g",model->rho[i]);
        fprintf(fp, "\n");
    }

    if(model->label)
    {
        fprintf(fp, "label");
        for(int i=0; i<nr_class; i++)
            fprintf(fp," %d",model->label[i]);
        fprintf(fp, "\n");
    }

    if(model->probA) // regression has probA only
    {
        fprintf(fp, "probA");
        for(int i=0; i<nr_class*(nr_class-1)/2; i++)
            fprintf(fp," %g",model->probA[i]);
        fprintf(fp, "\n");
    }
    if(model->probB)
    {
        fprintf(fp, "probB");
        for(int i=0; i<nr_class*(nr_class-1)/2; i++)
            fprintf(fp," %g",model->probB[i]);
        fprintf(fp, "\n");
    }

    if(model->nSV)
    {
        fprintf(fp, "nr_sv");
        for(int i=0; i<nr_class; i++)
            fprintf(fp," %d",model->nSV[i]);
        fprintf(fp, "\n");
    }

    fprintf(fp, "SV\n");
    const double * const *sv_coef = model->sv_coef;
    Xfloat **SV = model->SV;
    int **nz_sv = model->nz_sv;
    int *sv_len = model->sv_len;
    for(int i=0; i<l; i++)
    {
        for(int j=0; j<nr_class-1; j++)
            fprintf(fp, "%.16g ",sv_coef[j][i]);
        for(int k=0; k<sv_len[i]; ++k)
            fprintf(fp,"%d:%.8g ",nz_sv[i][k]+1,SV[i][k]);
        fprintf(fp, "\n");
    }
    fclose(fp);
    return 0;
}

svm_model *svm_load_model(const char *model_file_name)
{
    FILE *fp = fopen(model_file_name,"rb");
    if(fp==NULL) return NULL;

    // read parameters

    svm_model *model = Malloc(svm_model,1);
    svm_parameter& param = model->param;
    model->rho = NULL;
    model->probA = NULL;
    model->probB = NULL;
    model->label = NULL;
    model->nSV = NULL;
    int scanRet;
    #define CheckFscanf(n) if(n != scanRet){info("fscanf Err line %d, file %s: %d != %d\n",__LINE__,__FILE__,(n),scanRet);}
    char cmd[81];
    while(1)
    {
        scanRet = fscanf(fp,"%80s",cmd);CheckFscanf(1)

        if(strcmp(cmd,"svm_type")==0)
        {
            scanRet = fscanf(fp,"%80s",cmd);CheckFscanf(1)
            int i;
            for(i=0; svm_type_table[i]; i++)
            {
                if(strcmp(svm_type_table[i],cmd)==0)
                {
                    param.svm_type=i;
                    break;
                }
            }
            if(svm_type_table[i] == NULL)
            {
                fprintf(stderr,"unknown svm type.\n");
                free(model->rho);
                free(model->label);
                free(model->nSV);
                free(model);
                return NULL;
            }
        }
        else if(strcmp(cmd,"kernel_type")==0)
        {
            scanRet = fscanf(fp,"%80s",cmd);CheckFscanf(1);
            int i;
            for(i=0; kernel_type_table[i]; i++)
            {
                if(strcmp(kernel_type_table[i],cmd)==0)
                {
                    param.kernel_type=i;
                    break;
                }
            }
            if(kernel_type_table[i] == NULL)
            {
                fprintf(stderr,"unknown kernel function.\n");
                free(model->rho);
                free(model->label);
                free(model->nSV);
                free(model);
                return NULL;
            }
        }
        else if(strcmp(cmd,"degree")==0) {
            scanRet = fscanf(fp,"%d",&param.degree);
            CheckFscanf(1);
        } else if(strcmp(cmd,"gamma")==0) {
            scanRet = fscanf(fp,"%lf",&param.gamma);
            CheckFscanf(1);
        } else if(strcmp(cmd,"coef0")==0) {
            scanRet = fscanf(fp,"%lf",&param.coef0);
            CheckFscanf(1);
        } else if(strcmp(cmd,"nr_class")==0) {
            scanRet = fscanf(fp,"%d",&model->nr_class);
            CheckFscanf(1);
        } else if(strcmp(cmd,"total_sv")==0) {
            scanRet = fscanf(fp,"%d",&model->l);
            CheckFscanf(1);
        } else if(strcmp(cmd,"rho")==0)
        {
            int n = model->nr_class * (model->nr_class-1)/2;
            model->rho = Malloc(double,n);
            for(int i=0; i<n; i++) {
                scanRet = fscanf(fp,"%lf",&model->rho[i]);
                CheckFscanf(1);
            }
        }
        else if(strcmp(cmd,"label")==0)
        {
            int n = model->nr_class;
            model->label = Malloc(int,n);
            for(int i=0; i<n; i++) {
                scanRet = fscanf(fp,"%d",&model->label[i]);
                CheckFscanf(1);
            }
        }
        else if(strcmp(cmd,"probA")==0)
        {
            int n = model->nr_class * (model->nr_class-1)/2;
            model->probA = Malloc(double,n);
            for(int i=0; i<n; i++) {
                scanRet = fscanf(fp,"%lf",&model->probA[i]);
                CheckFscanf(1);
            }
        }
        else if(strcmp(cmd,"probB")==0)
        {
            int n = model->nr_class * (model->nr_class-1)/2;
            model->probB = Malloc(double,n);
            for(int i=0; i<n; i++) {
                scanRet = fscanf(fp,"%lf",&model->probB[i]);
                CheckFscanf(1);
            }
        }
        else if(strcmp(cmd,"nr_sv")==0)
        {
            int n = model->nr_class;
            model->nSV = Malloc(int,n);
            for(int i=0; i<n; i++) {
                scanRet = fscanf(fp,"%d",&model->nSV[i]);
                CheckFscanf(1);
            }
        }
        else if(strcmp(cmd,"SV")==0)
        {
            while(1)
            {
                int c = getc(fp);
                if(c==EOF || c=='\n') break;
            }
            break;
        }
        else
        {
            fprintf(stderr,"unknown text in model file\n");
            free(model->rho);
            free(model->label);
            free(model->nSV);
            free(model);
            return NULL;
        }
    }
	
    // read sv_coef and SV

    int elements = 0;
    long pos = ftell(fp);

    while(1)
    {
        int c = fgetc(fp);
        switch(c)
        {
        case '\n':
            // count the '-1' element
        case ':':
            ++elements;
            break;
        case EOF:
            goto out;
        default:
            ;
        }
    }
out:
    fseek(fp,pos,SEEK_SET);

    int m = model->nr_class - 1;
    int l = model->l;
    model->sv_coef = Malloc(double *,m);
    int i;
    for(i=0; i<m; i++)
        model->sv_coef[i] = Malloc(double,l);
    model->SV = Malloc(Xfloat *, l);
    model->nz_sv = Malloc(int *, l);
    model->sv_len = Malloc(int, l);
    Xfloat *x_space=NULL;
    int *nz_x_space=NULL;
    if(l>0)
    {
        x_space = Malloc(Xfloat,elements);
        nz_x_space = Malloc(int,elements);
    }
    model->max_idx = 0;
    int j=0;
    for(i=0; i<l; i++)
    {
        model->SV[i] = &x_space[j];
        model->nz_sv[i] = &nz_x_space[j];
        model->sv_len[i] = 0;
        for(int k=0; k<m; k++) {
            scanRet = fscanf(fp,"%lf",&model->sv_coef[k][i]);
            CheckFscanf(1);
        }
        while(1)
        {
            int c;
            do {
                c = getc(fp);
                if(c=='\n') goto out2;
            } while(isspace(c));
            ungetc(c,fp);
            //	  fscanf(fp,"%d:%lf",&nz_x_space[j],&x_space[j]);
            scanRet = fscanf(fp,"%d:%f",&nz_x_space[j],&x_space[j]);
            CheckFscanf(2);
            --nz_x_space[j]; // we need zero based indices
            ++model->sv_len[i];
            ++j;
        }
out2:
        if(j>=1 && nz_x_space[j-1]+1 > model->max_idx)
        {
            model->max_idx = nz_x_space[j-1]+1;
        }
    }
    #undef CheckFscanf
    fclose(fp);

    model->free_sv = 1;	// XXX
    return model;
}

void svm_destroy_model(svm_model* model)
{
    if(model->free_sv && model->l > 0)
    {
        free((void *)(model->SV[0]));
        free((void *)(model->nz_sv[0]));
    }
    for(int i=0; i<model->nr_class-1; i++)
        free(model->sv_coef[i]);
    free(model->SV);
    free(model->nz_sv);
    free(model->sv_len);
    free(model->sv_coef);
    free(model->rho);
    free(model->label);
    free(model->probA);
    free(model->probB);
    free(model->nSV);
    free(model);
}

void svm_destroy_param(svm_parameter* param)
{
    free(param->weight_label);
    free(param->weight);
}

const char *svm_check_parameter(const svm_problem *prob,
                                const svm_parameter *param)
{
    // svm_type

    int svm_type = param->svm_type;
    if(svm_type != C_SVC &&
            svm_type != NU_SVC &&
            svm_type != ONE_CLASS &&
            svm_type != EPSILON_SVR &&
            svm_type != NU_SVR)
        return "unknown svm type";

    // kernel_type

    int kernel_type = param->kernel_type;
    if(kernel_type != LINEAR &&
            kernel_type != POLY &&
            kernel_type != RBF &&
            kernel_type != SIGMOID)
        return "unknown kernel type";

    // cache_size,eps,C,nu,p,shrinking

    if(param->cache_size <= 0)
        return "cache_size <= 0";

    if(param->eps <= 0)
        return "eps <= 0";

    if(svm_type == C_SVC ||
            svm_type == EPSILON_SVR ||
            svm_type == NU_SVR)
        if(param->C <= 0)
            return "C <= 0";

    if(svm_type == NU_SVC ||
            svm_type == ONE_CLASS ||
            svm_type == NU_SVR)
        if(param->nu < 0 || param->nu > 1)
            return "nu < 0 or nu > 1";

    if(svm_type == EPSILON_SVR)
        if(param->p < 0)
            return "p < 0";

    if(param->shrinking != 0 &&
            param->shrinking != 1)
        return "shrinking != 0 and shrinking != 1";

    if(param->probability != 0 &&
            param->probability != 1)
        return "probability != 0 and probability != 1";

    if(param->probability == 1 &&
            svm_type == ONE_CLASS)
        return "one-class SVM probability output not supported yet";


    // check whether nu-svc is feasible

    if(svm_type == NU_SVC)
    {
        int l = prob->l;
        int max_nr_class = 16;
        int nr_class = 0;
        int *label = Malloc(int,max_nr_class);
        int *count = Malloc(int,max_nr_class);

        int i;
        for(i=0; i<l; i++)
        {
            int this_label = (int)prob->y[i];
            int j;
            for(j=0; j<nr_class; j++)
                if(this_label == label[j])
                {
                    ++count[j];
                    break;
                }
            if(j == nr_class)
            {
                if(nr_class == max_nr_class)
                {
                    max_nr_class *= 2;
                    label = (int *)realloc(label,max_nr_class*sizeof(int));
                    count = (int *)realloc(count,max_nr_class*sizeof(int));
                }
                label[nr_class] = this_label;
                count[nr_class] = 1;
                ++nr_class;
            }
        }

        for(i=0; i<nr_class; i++)
        {
            int n1 = count[i];
            for(int j=i+1; j<nr_class; j++)
            {
                int n2 = count[j];
                if(param->nu*(n1+n2)/2 > min(n1,n2))
                {
                    free(label);
                    free(count);
                    return "specified nu is infeasible";
                }
            }
        }
        free(label);
        free(count);
    }

    return NULL;
}

int svm_check_probability_model(const svm_model *model)
{
    return ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
            model->probA!=NULL && model->probB!=NULL) ||
           ((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
            model->probA!=NULL);
}

