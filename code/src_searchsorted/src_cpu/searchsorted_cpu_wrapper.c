#include <TH/TH.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

void searchsorted_cpu(float *res, float *a, float *v, int nrow_res, int nrow_a, int nrow_v, int ncol_a, int ncol_v);

void searchsorted_cpu_wrapper(THFloatTensor *a_tensor, THFloatTensor *v_tensor, THFloatTensor *res_tensor)
{
    // Get the dimensions
    long int nrow_a = THFloatTensor_size(a_tensor, 0);
    long int nrow_v = THFloatTensor_size(v_tensor, 0);
    int ncol_a = THFloatTensor_size(a_tensor, 1);
    int ncol_v = THFloatTensor_size(v_tensor, 1);

    // identify the number of rows for the result
    int nrow_res = fmax(nrow_a, nrow_v);

    // get the data of all tensors
    float *res = THFloatTensor_data(res_tensor);
    float *a = THFloatTensor_data(a_tensor);
    float *v = THFloatTensor_data(v_tensor);

    // launch the Float searchsorted function
    searchsorted_cpu(res, a, v, nrow_res, nrow_a, nrow_v, ncol_a, ncol_v);
}


int eval(float val, float *a, int row, int col, int ncol)
{
  /* Evaluates whether a[row,col] < val <= a[row, col+1]*/

    if (col == ncol-1){
      // we are on the right border. This is the answer.
      return 0;}

    // a[row,col] <= val ?
    int is_lower = (a[row*ncol + col] < val);

    // a[row,col+1] > val ?
    int is_next_higher = (a[row*ncol + col + 1] >= val);

    if (is_lower && is_next_higher) {
      // we found the answer
        return 0;
    } else if (is_lower) {
      // answer is on the right side
        return 1;
    } else {
      // answer is on the left side
        return -1;
    }
}

int binary_search(float *a, int row, float val, int ncol)
{
  /* Look for the value `val` within row `row` of matrix `a`, which
  has `ncol` columns.

  the `a` matrix is assumed sorted in increasing order, row-wise

  Returns -1 if `val` is smaller than the smallest value found within that
  row of `a`. Otherwise, return the column index `res` such that:
  a[row, col] < val <= a[row, col+1]. in case `val` is larger than the
  largest element of that row of `a`, simply return `ncol`-1. */

  //start with left at 0 and right at ncol
  int right = ncol;
  int left = 0;

  while (right >= left) {
      // take the midpoint of current left and right cursors
      int mid = left + (right-left)/2;

      // check the relative position of val: is this midpoint smaller or larger
      // than val ?
      int rel_pos = eval(val, a, row, mid, ncol);

      // we found the point
      if(rel_pos == 0) {
          return mid;
      } else if (rel_pos > 0) {
        // the answer is on the right side
          left = mid;
      } else {
        // the answer is on the left side
        if (!mid)
        {
          //if we're already on the first element, we didn't find
          return -1;}
        else
        {right = mid;}
      }
  }
  return -1;
}

void searchsorted_cpu(float *res, float *a, float *v, int nrow_res, int nrow_a, int nrow_v, int ncol_a, int ncol_v)
{
  for (int row=0; row<nrow_res; row++){
    for (int col=0; col<ncol_v; col++){
      // get the value to look for
      int row_in_v = (nrow_v==1) ? 0: row;
      int row_in_a = (nrow_a==1) ? 0: row;
      int idx_in_v = row_in_v*ncol_v+col;
      int idx_in_res = row*ncol_v+col;

      // apply binary search
      res[idx_in_res] = binary_search(a, row_in_a, v[idx_in_v], ncol_a)+1;
  }}
}
