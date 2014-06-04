#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "svm.h"

char* line;
int max_line_len = 1024;
Xfloat *x;
int *nz_x;
int max_nr_attr = 64;

struct svm_model* model;
int predict_probability=0;

// Determines the number of input patterns.
int num_patterns(FILE *input)
{
  int l=0;
  char c;
  do {
    c = getc(input);
    if(c=='\n')
      ++l;
  } while(c!=EOF);
  rewind(input);
  return l;
}

void setup_range(int *range_low, int *range_up, int total_sz, int size)
{
  int local_sz = total_sz/size;
  int idx_up = local_sz;
  int idx_low = 0;
  if(total_sz != 0)
    {
      for(int i=0; i<size-1; ++i)
	{
	  range_low[i] = idx_low;
	  range_up[i] = idx_up;
	  idx_low = idx_up;
	  idx_up = idx_low + local_sz + 1;
	}
      range_low[size-1] = idx_low; range_up[size-1]=total_sz;
    }
  else
    {
      for(int i=0; i<size; ++i)
	{
	  range_low[i] = 0;
	  range_up[i] = 0;
	}
    }
}

void predict_parallel(FILE *input, FILE *output, int l, MPI_Comm comm)
{
  int rank, size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  // Determine local range
  int l_low_loc, l_up_loc;
  int local_l = 0;
  int correct = 0; int other_correct;
  int total = 0;
  int ierr = 0;
  double error = 0; double other_error;
  double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
  double other_sumv = 0, other_sumy = 0, other_sumvv = 0, other_sumyy = 0;
  double other_sumvy = 0;
  int svm_type=svm_get_svm_type(model);
  int nr_class=svm_get_nr_class(model);
  double *prob_estimates=NULL;
  double *v;
  double target;
  int j,k,jj;
  int *l_up = (int *) malloc(size*sizeof(int));
  int *l_low = (int *) malloc(size*sizeof(int));

  setup_range(l_low, l_up, l, size);
  l_low_loc = l_low[rank];
  l_up_loc = l_up[rank];
  local_l = l_up_loc - l_low_loc;
  v = (double *) malloc((local_l+(size-1))*sizeof(double));

  if(predict_probability)
    {
      if(!(svm_type==NU_SVR || svm_type==EPSILON_SVR))
	{
	  prob_estimates = (double *) malloc(((local_l+(size-1))*nr_class)*
					     sizeof(double));
	}
    }
  k=0;
  while(1)
    {
      int i = 0;
      int c;

      if (fscanf(input,"%lf",&target)==EOF)
	break;
      while(1)
	{
	  if(i>=max_nr_attr-1)	// need one more for index = -1
	    {
	      max_nr_attr *= 2;
	      x = (Xfloat *) realloc(x,max_nr_attr*sizeof(double));
	      nz_x = (int *) realloc(nz_x,max_nr_attr*sizeof(int));
	    }

	  do {
	    c = getc(input);
	    if(c=='\n' || c==EOF) goto out2;
	  } while(isspace(c));
	  ungetc(c,input);
	  //	  fscanf(input,"%d:%lf",&nz_x[i],&x[i]);
	  fscanf(input,"%d:%f",&nz_x[i],&x[i]);
	  --nz_x[i]; // we need zero based indices
	  ++i;
	}	
    out2:
      if(l_low_loc <= total && total < l_up_loc)
	{
	  if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))
	    {
	      v[k] = svm_predict_probability(model, x, nz_x, i,
					     &prob_estimates[k*nr_class]);
	    }
	  else
	    {
	      v[k] = svm_predict(model, x, nz_x, i);
	    }
	  if(v[k] == target)
	    ++correct;
	  error += (v[k]-target)*(v[k]-target);
	  sumv += v[k];
	  sumy += target;
	  sumvv += v[k]*v[k];
	  sumyy += target*target;
	  sumvy += v[k]*target;
	  ++k;
	}
      ++total;
    }

  // Send all predictions to first processor, first processor
  // writes output file
  if(rank == 0)
    {
      if(predict_probability)
	{
	  if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
	    {
	      printf("Prob. model for test data: target value = predicted");
	      printf("value + z,\nz: Laplace distribution e^(-|z|/sigma)/");
	      printf("2sigma),sigma=%g\n",svm_get_svr_probability(model));
	    }
	  else
	    {
	      int *labels=(int *) malloc(nr_class*sizeof(int));
	      svm_get_labels(model,labels);
	      fprintf(output,"labels");		
	      for(j=0;j<nr_class;j++)
		fprintf(output," %d",labels[j]);
	      fprintf(output,"\n");
	      free(labels);
	    }
	}
      for(k=0; k<local_l; ++k)
	if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))
	  {
	    fprintf(output,"%g ",v[k]);
	    for(j=0;j<nr_class;j++)
	      fprintf(output,"%g ",prob_estimates[k*nr_class+j]);
	    fprintf(output,"\n");
	  }
	else
	  {
	    fprintf(output,"%g\n",v[k]);
	  }
      for(j=1; j<size; ++j){
	MPI_Status stat;
	ierr = MPI_Recv(v, l_up[j]-l_low[j], MPI_DOUBLE, 
			j, 0, comm, &stat);
	ierr = MPI_Recv(&other_correct, 1, MPI_INT, j, 0, comm, &stat);
	correct += other_correct;
	ierr = MPI_Recv(&other_sumv, 1, MPI_DOUBLE, j, 0, comm, &stat);
	sumv += other_sumv;
	ierr = MPI_Recv(&other_sumy, 1, MPI_DOUBLE, j, 0, comm, &stat);
	sumy += other_sumy;
	ierr = MPI_Recv(&other_sumvy, 1, MPI_DOUBLE, j, 0, comm, &stat);
	sumvy += other_sumvy;
	ierr = MPI_Recv(&other_sumvv, 1, MPI_DOUBLE, j, 0, comm, &stat);
	sumvv += other_sumvv;
	ierr = MPI_Recv(&other_sumyy, 1, MPI_DOUBLE, j, 0, comm, &stat);
	sumyy += other_sumyy;
	ierr = MPI_Recv(&other_error, 1, MPI_DOUBLE, j, 0, comm, &stat);
	error += other_error;
	if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))
	  {
	    ierr = MPI_Recv(prob_estimates, (l_up[j]-l_low[j])*nr_class, 
			    MPI_DOUBLE, j, 0, comm, &stat);
	    for(k=0; k<l_up[j]-l_low[j]; ++k)
	      {
		fprintf(output,"%g ",v[k]);
		for(jj=0;jj<nr_class;jj++)
		  fprintf(output,"%g ",prob_estimates[k*nr_class+jj]);
		fprintf(output,"\n");
	      }
	  }
	else
	  {
	    for(k=0; k<l_up[j]-l_low[j]; ++k)
	      {
		fprintf(output,"%g\n",v[k]);
	      }
	  }
      }
      printf("Accuracy = %g%% (%d/%d) (classification)\n",
	     (double)correct/total*100,correct,total);
      printf("Mean squared error = %g (regression)\n",error/total);
      printf("Squared correlation coefficient = %g (regression)\n",
	     ((total*sumvy-sumv*sumy)*(total*sumvy-sumv*sumy))/
	     ((total*sumvv-sumv*sumv)*(total*sumyy-sumy*sumy))
	     );
    } // if(rank == 0)
  else
    {
      ierr = MPI_Send(v, local_l, MPI_DOUBLE, 0, 0, comm);
      ierr = MPI_Send(&correct, 1, MPI_INT, 0, 0, comm);
      ierr = MPI_Send(&sumv, 1, MPI_DOUBLE, 0, 0, comm);
      ierr = MPI_Send(&sumy, 1, MPI_DOUBLE, 0, 0, comm);
      ierr = MPI_Send(&sumvy, 1, MPI_DOUBLE, 0, 0, comm);
      ierr = MPI_Send(&sumvv, 1, MPI_DOUBLE, 0, 0, comm);
      ierr = MPI_Send(&sumyy, 1, MPI_DOUBLE, 0, 0, comm);
      ierr = MPI_Send(&error, 1, MPI_DOUBLE, 0, 0, comm);

      if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))
	{
	  ierr = MPI_Send(prob_estimates, local_l*nr_class, 
			  MPI_DOUBLE, 0, 0, comm);
	}
    }

  if(predict_probability)
    {
      free(prob_estimates);
    }
  free(v);
  free(l_up);
  free(l_low);
}

void exit_with_help()
{
  printf(
	 "Usage: svm-predict [options] test_file model_file output_file\n"
	 "options:\n"
	 "-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); one-class SVM not supported yet\n"
	 );
  exit(1);
}

int main(int argc, char **argv)
{
  FILE *input, *output;
  int i;

  MPI_Init(&argc, &argv);

  // parse options
  for(i=1;i<argc;i++)
    {
      if(argv[i][0] != '-') break;
      ++i;
      switch(argv[i-1][1])
	{
	case 'b':
	  predict_probability = atoi(argv[i]);
	  break;
	default:
	  fprintf(stderr,"unknown option\n");
	  exit_with_help();
	}
    }
  if(i>=argc)
    exit_with_help();
	
  input = fopen(argv[i],"r");
  if(input == NULL)
    {
      fprintf(stderr,"can't open input file %s\n",argv[i]);
      exit(1);
    }

  output = fopen(argv[i+2],"w");
  if(output == NULL)
    {
      fprintf(stderr,"can't open output file %s\n",argv[i+2]);
      exit(1);
    }

  if((model=svm_load_model(argv[i+1]))==0)
    {
      fprintf(stderr,"can't open model file %s\n",argv[i+1]);
      exit(1);
    }
	
  line = (char *) malloc(max_line_len*sizeof(char));
  x = (Xfloat *) malloc(max_nr_attr*sizeof(Xfloat));
  nz_x = (int *) malloc(max_nr_attr*sizeof(int));
  if(predict_probability)
    if(svm_check_probability_model(model)==0)
      {
	fprintf(stderr,"model does not support probabiliy estimates\n");
	predict_probability=0;
      }
  int l = num_patterns(input);
  //  predict(input,output,l);
  predict_parallel(input, output, l, MPI_COMM_WORLD);
  svm_destroy_model(model);
  free(line);
  free(x);
  free(nz_x);
  fclose(input);
  fclose(output);
  return 0;
}
