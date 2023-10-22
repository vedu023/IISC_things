#include<stdio.h>
#include<stdlib.h>

struct node
{
    int data;
    struct node *next;
};

typedef struct node node;

void push(node **head, int data)
{
     node* new_node = (node*)malloc(sizeof(node*));
     new_node->data = data;
     new_node->next = *head;
     *head = new_node;
}

void reverse(node **head)
{
     node *current = *head; 
     node *prev = NULL, *next = NULL; 
  
  
        while (current != NULL) 
        {
            next = current->next; 
            current->next = prev; 
  
            prev = current; 
            current = next; 
        } 
        *head = prev; 
}

node* addition(node* num1, node* num2)
{
  node* result = NULL;
  int sum, c = 0;
  
  while(num1 != NULL || num2 != NULL)
  {
        if(num1 && num2)
        {
            sum = c + num1->data + num2->data;
            num1 = num1->next;
            num2 = num2->next;
        }    
        else if(num1)
        {
            sum = c + num1->data;
            num1 = num1->next;
        }
        else if(num2)
        {
            sum = c + num2->data;
            num2 = num2->next;
        }
        else 
            sum = c;

        c = sum / 1000;
        sum = sum % 1000;
        push(&result, sum);    
  } 
  reverse(&result);
  return result;
}


void display(node *node)
{
    reverse(&node);
    struct node *temp = node;

    while(temp!= NULL)
    {
         printf("%03d", temp->data);
         temp = temp->next;
    }
 
    printf("%03d$\n",temp->data);
}

node* multiplication(node *num1, node *num2)
{

     node *temp = NULL; 
     node *result = NULL;
     int k = 0, i;
     
     while(num2)
     { 
           int sum = 0, c = 0;
           node *temp1 = num1;
           while(temp1)
           {
                sum = (c + (temp1->data) * (num2->data)) % 1000;
                c = (c + (temp1->data) * (num2->data)) / 1000;
                
                push(&temp, sum);
                temp1 = temp1->next;
           }
           
           if(c != 0)
               push(&temp, c);
             
           reverse(&temp);
           for (i=0; i<k; i++)
           {
                push(&temp, 0);
           } 
           
           k += 1;
           result = addition(temp, result);  
           temp = NULL;
           num2 = num2->next;
     }
     return result;
}


node* operands(char * ch, int i){
  node *op;
  int n;
  while(1)
  {
    char num[4] = {'\0'};
    for(int j = 0; j < 3; j++){
      num[j] = ch[i++];
    }
    n = atoi(num);
    node* n_node = (node*)malloc(sizeof(node*));
    n_node->data = n;
    n_node->next = op;
    op = n_node;

    if(ch[i] == ','){
      i++;
    }
    else if (ch[i] == '$'){
      return op;
    }
    else{
      printf("not given format");
      return NULL;
    }
  }
  return NULL;
}


void cal(char *ch){
  int i = 0;
  char operator= 'c';
  
  node *acc;
  node *operand;
  acc = operands(ch, 0);
  display(acc);

  while(ch[i++] != '$');
  i++;

  while (1){

    if (ch[i] == '='){
      display(acc);
    }
    else if (ch[i] == '+'){
      operator = '+';
      i++;
    }
    else if (ch[i] == '*'){
      operator = '*';
      i++;
    }
    else{
      printf("wrong operator");
    }

    i++;

    operand = operands(ch,i);
    while(ch[i++] != '$');
    i++;

    if (operator == '+'){
      acc = addition(acc, operand);
    }
    else if (ch[i] == '*'){
      acc = multiplication(acc, operand);
    }
  }

}
int main(){

    while(1){
      char *line = NULL;
      size_t len = 0;
      getline(&line, &len, stdin);
      if (line[0] == '='){
        break;
      }
      cal(line);
    }

}
