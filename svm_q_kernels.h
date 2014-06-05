#ifndef SVM_Q_KERNELS_INCLUDED
#define SVM_Q_KERNELS_INCLUDED

//
// Q matrices for various formulations
//
class SVC_Q: public Kernel
{
public:
    SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_)
        :Kernel(prob.l, prob.x, prob.nz_idx, prob.x_len, prob.max_idx, param)
    {
        clone(y,y_,prob.l);
        this->l = prob.l;
        cache = new Cache(prob.l,(int)(param.cache_size*(1<<20)));
        QD = new Qfloat[prob.l];
        for(int i=0; i<prob.l; i++)
            QD[i]= (Qfloat)(this->*kernel_function)(i,i);
    }

    Qfloat *get_Q(int column, int len) const
    {
        Qfloat *data;
        int start;
        if((start = cache->get_data(column,&data,len)) < len)
        {
            for(int j=start; j<len; j++)
                data[j] = (Qfloat)(y[column]*y[j]*(this->*kernel_function)(column,j));
        }
        return data;
    }

    Qfloat *get_QD() const
    {
        return QD;
    }

    Qfloat *get_Q_subset(int i, int *idxs, int n) const
    {
        Qfloat *data;
        int start = cache->get_data(i,&data,l);
        if(start == 0) // Initialize cache row
        {
            for(int j=0; j<l; ++j)
                data[j] = NAN;
        }
        for(int j=0; j<n; ++j)
        {
            if(isnan(data[idxs[j]]))
                data[idxs[j]] = (Qfloat)(y[i]*y[idxs[j]]*
                                         (this->*kernel_function)(i,idxs[j]));
        }
        return data;
    }

    Qfloat get_non_cached(int i, int j) const
    {
        return (Qfloat) y[i]*y[j]*(this->*kernel_function)(i,j);
    }

    inline bool is_cached(const int i) const
    {
        return cache->is_cached(i);
    }

    void swap_index(int i, int j) const
    {
        cache->swap_index(i,j);
        Kernel::swap_index(i,j);
        swap(y[i],y[j]);
        swap(QD[i],QD[j]);
    }

    ~SVC_Q()
    {
        delete[] y;
        delete cache;
        delete[] QD;
    }
protected:
    schar *y;
    Cache *cache;
    Qfloat *QD;
    int l;
};


class ONE_CLASS_Q: public Kernel
{
public:
    ONE_CLASS_Q(const svm_problem& prob, const svm_parameter& param)
        :Kernel(prob.l, prob.x, prob.nz_idx, prob.x_len, prob.max_idx, param)
    {
        this->l = prob.l;
        cache = new Cache(prob.l,(int)(param.cache_size*(1<<20)));
        QD = new Qfloat[prob.l];
        for(int i=0; i<prob.l; i++)
            QD[i]= (Qfloat)(this->*kernel_function)(i,i);
    }

    Qfloat *get_Q(int column, int len) const
    {
        Qfloat *data;
        int start;
        if((start = cache->get_data(column,&data,len)) < len)
        {
            for(int j=start; j<len; j++)
                data[j] = (Qfloat)(this->*kernel_function)(column,j);
        }
        return data;
    }

    Qfloat *get_QD() const
    {
        return QD;
    }

    Qfloat *get_Q_subset(int i, int *idxs, int n) const
    {
        Qfloat *data;
        int start = cache->get_data(i,&data,l);
        if(start == 0) // Initialize cache row
        {
            for(int j=0; j<l; ++j)
                data[j] = NAN;
        }
        for(int j=0; j<n; ++j)
        {
            if(isnan(data[idxs[j]]))
                data[idxs[j]] = (Qfloat)(this->*kernel_function)(i,idxs[j]);
        }
        return data;
    }

    Qfloat get_non_cached(int i, int j) const
    {
        return (Qfloat) (this->*kernel_function)(i,j);
    }

    inline bool is_cached(const int i) const
    {
        return cache->is_cached(i);
    }

    void swap_index(int i, int j) const
    {
        cache->swap_index(i,j);
        Kernel::swap_index(i,j);
        swap(QD[i],QD[j]);
    }

    ~ONE_CLASS_Q()
    {
        delete cache;
        delete[] QD;
    }
private:
    Cache *cache;
    Qfloat *QD;
    int l;
};

class SVR_Q: public Kernel
{
public:
    SVR_Q(const svm_problem& prob, const svm_parameter& param)
        :Kernel(prob.l, prob.x, prob.nz_idx, prob.x_len, prob.max_idx, param)
    {
        l = prob.l;
        cache = new Cache(l,(int)(param.cache_size*(1<<20)));
        QD = new Qfloat[2*l];
        sign = new schar[2*l];
        index = new int[2*l];
        for(int k=0; k<l; k++)
        {
            sign[k] = 1;
            sign[k+l] = -1;
            index[k] = k;
            index[k+l] = k;
            QD[k]= (Qfloat)(this->*kernel_function)(k,k);
            QD[k+l]=QD[k];
        }
        buffer[0] = new Qfloat[2*l];
        buffer[1] = new Qfloat[2*l];
        next_buffer = 0;
    }

    void swap_index(int i, int j) const
    {
        swap(sign[i],sign[j]);
        swap(index[i],index[j]);
        swap(QD[i],QD[j]);
    }

    Qfloat *get_Q(int column, int len) const
    {
        Qfloat *data;
        int real_column = index[column];
        if(cache->get_data(real_column,&data,l) < l)
        {
            for(int j=0; j<l; j++)
                data[j] = (Qfloat)(this->*kernel_function)(real_column,j);
        }

        // reorder and copy
        Qfloat *buf = buffer[next_buffer];
        next_buffer = 1 - next_buffer;
        schar si = sign[column];
        for(int j=0; j<len; j++)
            buf[j] = si * sign[j] * data[index[j]];
        return buf;
    }

    Qfloat *get_QD() const
    {
        return QD;
    }

    Qfloat *get_Q_subset(int i, int *idxs, int n) const
    {
        Qfloat *data;
        int real_i = index[i];
        int start = cache->get_data(real_i,&data,l);
        if(start == 0) // Initialize cache row
        {
            for(int j=0; j<l; ++j)
            {
                data[j] = NAN;
            }
        }
        for(int j=0; j<n; ++j)
        {
            int real_j = index[idxs[j]];
            if(isnan(data[real_j]))
                data[real_j] = (Qfloat)(this->*kernel_function)(real_i,real_j);
        }
        // reorder and copy
        Qfloat *buf = buffer[next_buffer];
        next_buffer = 1 - next_buffer;
        schar si = sign[i];
        for(int j=0; j<n; ++j)
            buf[idxs[j]] = si * sign[idxs[j]] * data[index[idxs[j]]];
        return buf;
    }

    Qfloat get_non_cached(int i, int j) const
    {
        int real_i = index[i];
        int real_j = index[j];
        return (Qfloat) sign[i]*sign[j]*(this->*kernel_function)(real_i,real_j);
    }

    inline bool is_cached(const int i) const
    {
        return cache->is_cached(i);
    }

    ~SVR_Q()
    {
        delete cache;
        delete[] sign;
        delete[] index;
        delete[] buffer[0];
        delete[] buffer[1];
        delete[] QD;
    }
protected:
    int l;
    Cache *cache;
    schar *sign;
    int *index;
    mutable int next_buffer;
    Qfloat *buffer[2];
    Qfloat *QD;
};

#endif
