#ifndef SVM_CACHE_INCLUDED
#define SVM_CACHE_INCLUDED
//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
#include "svm.h"

class Cache
{
public:
  Cache(int l,int size);
  ~Cache();

  // request data [0,len)
  // return some position p where [p,len) need to be filled
  // (p >= len if nothing needs to be filled)
  int get_data(const int index, Qfloat **data, int len);
  void swap_index(int i, int j);	// future_option
  // check if row is in cache
  bool is_cached(const int index) const;
private:
  int l;
  int size;
  struct head_t
  {
    head_t *prev, *next;	// a cicular list
    Qfloat *data;
    int len;		// data[0,len) is cached in this entry
  };

  head_t *head;
  head_t lru_head;
  void lru_delete(head_t *h);
  void lru_insert(head_t *h);
};
#endif
