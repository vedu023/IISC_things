

#include <stdio.h>

int main ()
{

  float f = 1.0;
  double d = 1.0;

  int c1 = 0, c2 = 0; 
  
//  single precision
  while (f != 0.0)
    {
      f = f / 2.0;
      c1++;
    }

// double precision
  while (d != 0.0)
    {
      d = d / 2.0;
      c2++;
    }

  printf ("steps for single precision...%d\n", c1);
  printf ("steps for double precision...%d\n", c2);
  return 0;
}
