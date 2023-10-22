
#include <stdio.h> 

void main()
{
    float n,i;
    
    printf("Enter the value of n(for single):... ");
    scanf("%f", &n);

    float s1 = 0.0, s2 = 0.0;
    double sd1 = 0.0, sd2 = 0.0;
    
    for (i = 1; i < n; i++)
    {
        s1 = s1 + 1 / i;
        s2 = s2 + 1 /(n-i);
    }
    
    printf("\n\n-------------for float IEEE single precision--------------------\n");
    printf("Sum upto %.0fth value in HP is: %f\n", n, s1);
    printf("Sum upto %.0fth value in HP of reverse order is: %f\n", n, s2);
    
    printf("error : %f\n", s1 - s2);
    
    
    for ( i = 1; i < n; i++)
    {
        sd1 = sd1 + 1 / i;
        sd2 = sd2 + 1 /(n-i);
    }
    
    printf("\n--------------for IEEE double precision-------------------------\n ");
    printf("Sum upto %.0fth value in HP is: %lf\n", n, sd1);
    printf("Sum upto %.0fth value in HP of reverse order is: %lf\n", n, sd2);
    
    printf("error : %f\n", sd1 - sd2);
}
