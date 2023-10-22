#include <stdio.h>
#include <stdlib.h>

typedef union data
{
   int d;
   float f;
}D;

void dec_bin(unsigned num);

void main()
{
    D d1,d2;
    d1.f = 22.0/7;  // pi_1 = 22/7
    d2.d = 0x40490FDB; //pi_2 
     
    printf("decimal value of pi_1....%f\n",d1.f);
    printf("decimal value of pi_2....%f\n",d2.f);
    
    printf("diff...%f\n",(d1.f-d2.f));
    
    printf("binary string_1...");
    dec_bin(d1.d);
    
    printf("\nbinary string_2...");
    dec_bin(d2.d);
    
    printf("\n");
}

void dec_bin (unsigned num)
{
   unsigned i = 1;
   i = i << 31;
   while(i!=0)
   {
     printf("%d",num & i? 1:0);
     i = i/2;
   }

}
