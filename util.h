// D. Brugger, december 2006
// $Id: util.h 573 2010-12-29 10:54:20Z dome $
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

#ifndef __UTIL__H__
#define __UTIL__H__

#include <stdlib.h>
#include <string.h>

#ifndef min
template <class T> inline T min(T x,T y) {
    return (x<y)?x:y;
}
#endif
#ifndef max
template <class T> inline T max(T x,T y) {
    return (x>y)?x:y;
}
#endif
template <class T> inline void swap(T& x, T& y) {
    T t=x;
    x=y;
    y=t;
}
template <class S, class T> inline void clone(T*& dst, S* src, int n)
{
    dst = new T[n];
    memcpy((void *)dst,(void *)src,sizeof(T)*n);
}

template <class T>
void quick_sort(T a[], int idxs[], int l_, int r_)
{
    int     i, j, s, d, l, iw, ps[20], pd[20];
    T  x, w;

    l     = 0;
    ps[0] = l_;
    pd[0] = r_;
    do
    {
        s = ps[l];
        d = pd[l];
        l--;
        do
        {
            i = s;
            j = d;
            x = a[(s+d)/2];
            do
            {
                while (a[i] < x) i++;
                while (a[j] > x) j--;
                if (i <= j)
                {
                    iw    = idxs[i];
                    w     = a[i];
                    idxs[i] = idxs[j];
                    a[i]  = a[j];
                    i++;
                    idxs[j] = iw;
                    a[j]  = w;
                    j--;
                }
            } while (i <= j);
            if (j-s > d-i)
            {
                l++;
                ps[l] = s;
                pd[l] = j;
                s     = i;
            }
            else
            {
                if (i < d)
                {
                    l++;
                    ps[l] = i;
                    pd[l] = d;
                }
                d = j;
            }
        } while (s < d);
    } while(l>=0);
}

#endif
