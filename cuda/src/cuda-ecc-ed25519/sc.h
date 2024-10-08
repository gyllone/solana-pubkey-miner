#ifndef SC_H
#define SC_H

/*
The set of scalars is \Z/l
where l = 2^252 + 27742317777372353535851937790883648493.
*/

__host__ __device__ void sc_reduce(unsigned char *s);
__host__ __device__ void sc_muladd(unsigned char *s, const unsigned char *a, const unsigned char *b, const unsigned char *c);

#endif
