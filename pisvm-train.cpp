#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <mpi.h>
#include <time.h>
#include "svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void exit_with_help()
{
    printf(
        "Usage: svm-train [options] training_set_file [model_file]\n"
        "options:\n"
        "-s svm_type : set type of SVM (default 0)\n"
        "	0 -- C-SVC\n"
        "	1 -- nu-SVC\n"
        "	2 -- one-class SVM\n"
        "	3 -- epsilon-SVR\n"
        "	4 -- nu-SVR\n"
        "-t kernel_type : set type of kernel function (default 2)\n"
        "	0 -- linear: u'*v\n"
        "	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
        "	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
        "	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
        "-d degree : set degree in kernel function (default 3)\n"
        "-g gamma : set gamma in kernel function (default 1/k)\n"
        "-r coef0 : set coef0 in kernel function (default 0)\n"
        "-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
        "-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
        "-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
        "-m cachesize : set cache memory size in MB (default 40)\n"
        "-e epsilon : set tolerance of termination criterion (default 0.001)\n"
        "-h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
        "-b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
        "-wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
        "-v n: n-fold cross validation mode\n"
        "-o n: max. size of working set\n"
        "-q n: max. number of new variables entering working set\n"
    );
    exit(1);
}

void parse_command_line(int argc, char **argv, char *input_file_name,
                        char *model_file_name);
void read_problem(const char *filename);
void do_cross_validation();

struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model;
Xfloat *x_space;
int *nz_idx_space;
int cross_validation;
int nr_fold;

int main(int argc, char **argv)
{
    char input_file_name[1024];
    char model_file_name[1024];
    const char *error_msg;
    double time = 0;
    MPI_Init(&argc, &argv);

    parse_command_line(argc, argv, input_file_name, model_file_name);
    time = MPI_Wtime();
    read_problem(input_file_name);
    time = MPI_Wtime() - time;
    error_msg = svm_check_parameter(&prob,&param);
    if(error_msg)
    {
        fprintf(stderr,"Error: %s\n",error_msg);
        exit(1);
    }

    if(cross_validation)
    {
        do_cross_validation();
    }
    else
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        model = svm_train(&prob,&param);
        if (rank == 0) svm_save_model(model_file_name,model);
        svm_destroy_model(model);
    }
    svm_destroy_param(&param);
    printf("I/O time = %.2lf\n", time);
    free(prob.y);
    free(prob.x);
    free(prob.nz_idx);
    free(prob.x_len);
    free(x_space);
    free(nz_idx_space);

    MPI_Finalize();

    return 0;
}

void do_cross_validation()
{
    int i;
    int total_correct = 0;
    double total_error = 0;
    double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
    double *target = Malloc(double,prob.l);

    svm_cross_validation(&prob,&param,nr_fold,target);
    if(param.svm_type == EPSILON_SVR ||
            param.svm_type == NU_SVR)
    {
        for(i=0; i<prob.l; i++)
        {
            double y = prob.y[i];
            double v = target[i];
            total_error += (v-y)*(v-y);
            sumv += v;
            sumy += y;
            sumvv += v*v;
            sumyy += y*y;
            sumvy += v*y;
        }
        printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
        printf("Cross Validation Squared correlation coefficient = %g\n",
               ((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
               ((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
              );
    }
    else
    {
        for(i=0; i<prob.l; i++)
            if(target[i] == prob.y[i])
                ++total_correct;
        printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
    }
    free(target);
}

void parse_command_line(int argc, char **argv, char *input_file_name,
                        char *model_file_name)
{
    int i;

    // default values
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.degree = 3;
    param.gamma = 0;	// 1/k
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 40;
    param.C = 1;
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    param.o = 2; // safe defaults
    param.q = 2;
    cross_validation = 0;

    // parse options
    for(i=1; i<argc; i++)
    {
        if(argv[i][0] != '-') break;
        if(++i>=argc)
            exit_with_help();
        switch(argv[i-1][1])
        {
        case 'o':
            param.o = atoi(argv[i]);
            break;
        case 'q':
            param.q = atoi(argv[i]);
            break;
        case 's':
            param.svm_type = atoi(argv[i]);
            break;
        case 't':
            param.kernel_type = atoi(argv[i]);
            break;
        case 'd':
            param.degree = atoi(argv[i]);
            break;
        case 'g':
            param.gamma = atof(argv[i]);
            break;
        case 'r':
            param.coef0 = atof(argv[i]);
            break;
        case 'n':
            param.nu = atof(argv[i]);
            break;
        case 'm':
            param.cache_size = atof(argv[i]);
            break;
        case 'c':
            param.C = atof(argv[i]);
            break;
        case 'e':
            param.eps = atof(argv[i]);
            break;
        case 'p':
            param.p = atof(argv[i]);
            break;
        case 'h':
            param.shrinking = atoi(argv[i]);
            break;
        case 'b':
            param.probability = atoi(argv[i]);
            break;
        case 'v':
            cross_validation = 1;
            nr_fold = atoi(argv[i]);
            if(nr_fold < 2)
            {
                fprintf(stderr,"n-fold cross validation: n must >= 2\n");
                exit_with_help();
            }
            break;
        case 'w':
            ++param.nr_weight;
            param.weight_label =
                (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
            param.weight =
                (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
            param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
            param.weight[param.nr_weight-1] = atof(argv[i]);
            break;
        default:
            fprintf(stderr,"unknown option\n");
            exit_with_help();
        }
    }

    // determine filenames

    if(i>=argc)
        exit_with_help();

    strcpy(input_file_name, argv[i]);

    if(i == argc - 2) //There is exactly 1 additional parameter after the input file name
        strcpy(model_file_name,argv[i+1]);
    else if (i == argc - 1)//input file name is the last parameter
    {
        char *p = strrchr(argv[i],'/');
        if(p==NULL)
            p = argv[i];
        else
            ++p;
        sprintf(model_file_name,"%s.model",p);
    } else { //There are more parameters
        printf("ERROR: There are unparsed parameters!\n");
        exit_with_help();
    }
}

// read in a problem (in svmlight format)

void read_problem(const char *filename)
{
    int elements, i, j;
    FILE *fp = fopen(filename,"r");

    if(fp == NULL)
    {
        fprintf(stderr,"can't open input file %s\n",filename);
        exit(1);
    }

    prob.l = 0;
    elements = 0;
    while(1)
    {
        int c = fgetc(fp);
        switch(c)
        {
        case '\n':
            ++prob.l;
            break;
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
    rewind(fp);
    prob.y = Malloc(double,prob.l);
    prob.x = Malloc(Xfloat *, prob.l);
    prob.nz_idx = Malloc(int *, prob.l);
    prob.x_len = Malloc(int, prob.l);
    //TODO: Check if not needed - loop sets prob.x_len[i] = 0 for i=0 to prob.l
    memset(prob.x_len, 0, sizeof(int)*prob.l);
    x_space = Malloc(Xfloat,elements);
    nz_idx_space = Malloc(int,elements);

    prob.max_idx = 0;
    j=0;
    for(i=0; i<prob.l; i++)
    {
        double label;
        prob.x[i] = &x_space[j];
        prob.nz_idx[i] = &nz_idx_space[j];
        prob.x_len[i] = 0;
        fscanf(fp,"%lf",&label);
        prob.y[i] = label;
        while(1)
        {
            int c;
            do {
                c = getc(fp);
                if(c=='\n') goto out2;
            } while(isspace(c));
            ungetc(c,fp);
            //	  fscanf(fp,"%d:%lf",&nz_idx_space[j],&x_space[j]);
            fscanf(fp,"%d:%f",&nz_idx_space[j],&x_space[j]);
            --nz_idx_space[j]; // we need zero based indices
            ++prob.x_len[i];
            ++j;
        }
out2:
        if(j>=1 && nz_idx_space[j-1]+1 > prob.max_idx)
        {
            prob.max_idx = nz_idx_space[j-1]+1;
        }
    }
    if(param.gamma == 0)
        param.gamma = 1.0/prob.max_idx;

    fclose(fp);
}
