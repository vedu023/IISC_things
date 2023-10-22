#include<stdio.h>
#include<stdlib.h>
#include<string.h>


struct CNode
{
	int data;
    struct CNode *link;
};

struct BTnode
{
	char *word;
	struct CNode *count;
    struct BTnode *left,*right;        
};

typedef struct CNode cnode;
typedef struct BTnode tnode;

cnode* c_newnode(int lno)
{
	cnode *new_node = (cnode *)malloc(sizeof(cnode));
	new_node->data = lno;
	new_node->link = NULL;
	return new_node;
}

cnode* c_insert(cnode *head, int lno)
{
	cnode *ptr = head;
	if(!head)
		head = c_newnode(lno);
	else 
	{
		while(ptr->link != NULL)
			ptr = ptr->link;
		ptr->link = c_newnode(lno);
	}
	return head;
}

void c_unique(cnode *root)
{
    cnode *ptr1, *ptr2, *dup;
    ptr1 = root;
 
    while (ptr1 != NULL && ptr1->link != NULL) 
    {
        ptr2 = ptr1;
        while (ptr2->link != NULL) 
        {
            if (ptr1->data == ptr2->link->data) 
            {
                dup = ptr2->link;
                ptr2->link = ptr2->link->link;
                free(dup);
            }
            else 
                ptr2 = ptr2->link;
        }
        ptr1 = ptr1->link;
    }
}

void print_list(cnode *head)
{
	if(!head)
		printf("empty list...");
	while(head != NULL)
	{
		printf("%d,",head->data);
		head = head->link;
	}
}

tnode* t_newnode(char *word, int lno)
{
	int s;
	tnode *new_node = (tnode*)malloc(sizeof(tnode));

	s = strlen(word)+1;
	new_node->word = (char *)malloc(sizeof(char)*s);
	strcpy(new_node->word, word);
	new_node->count = c_newnode(lno);
    new_node->left = NULL;
    new_node->right = NULL;
    return new_node;
}

tnode* t_insert(tnode* root, char *word, int lno)
{
	if(!root)
	{
        tnode* node = t_newnode(word,lno);
        return node;
    }

    if(strcmp(word,root->word) == 0)
    	root->count = c_insert(root->count,lno);

    else if(strcmp(word,root->word) < 0)
    	root->left = t_insert(root->left,word,lno);

    else if(strcmp(word,root->word) > 0)
    	root->right = t_insert(root->right,word,lno);

    return root;
}

tnode* t_min(tnode *root)
{
	if(!root)
		return NULL;
    while(root->left)
    	root = root->left;
    return root;
}

void swapStr(char **str1_ptr, char **str2_ptr)
{
  char *temp = *str1_ptr;
  *str1_ptr = *str2_ptr;
  *str2_ptr = temp;
} 

void c_swap(cnode** a, cnode** b)
{
  cnode *temp=*a;
  *a=*b;
  *b=temp;
}


tnode* t_delete(tnode *root, char *word)
{
        if(!root)
        	return NULL;
        
        if(strcmp(word,root->word)<0)
            root->left = t_delete(root->left,word);

        else if(strcmp(word,root->word)>0)
            root->right = t_delete(root->right,word);
        
        else
        {
            if(!root->right)
                return root->left;

            else if(!root->left)
                return root->right;
            
            else
            {
                tnode* tmin = t_min(root->right);
                swapStr(&root->word,&tmin->word);
                c_swap(&root->count,&tmin->count);
                root->right = t_delete(root->right,tmin->word);
                return root;
            }
        }
        
        return root;
}

char* w_purify(char *string)
{
    int s,i=0;
    s = strlen(string);

    for(i=0; i<s; i++)
    {
    	if(string[i]== '-')
    		string[i+1] = string[i];
    }
    return string;
}

void inorder(tnode* root)
{
  if(!root)
  	return;

  inorder(root->left);
  printf("%s : ",root->word);
  c_unique(root->count);
  print_list(root->count);
  printf("\n");
  inorder(root->right);
}

int l_count(cnode* root)
{
	int count = 0;
	if(!root)
		return 0;
	while(root)
	{
		root = root->link;
		count += 1;
	}
	return count;
}

tnode* reduce_tree(tnode *root)
{
  if(root==NULL)
  	return NULL;

  root->left = reduce_tree(root->left);
  root->right = reduce_tree(root->right);

  if(strlen(root->word)<3 || l_count(root->count)<3)
  	 root = t_delete(root,root->word);
  
  return root;
}


int main()
{
	FILE *f;
	tnode *root = NULL;
	int lno = 1;
	char str[1500];

	f = fopen("inputf.txt","r");

	if(!f)
		printf("ERROR :: file not open...");

	else
	{
		while(fgets(str, 1000, f)!= NULL)
		{   
			if (str[0] != 13)
			{
                char * token = strtok(str, " ");
                while( token != NULL ) 
                {
                	root = t_insert(root,token,lno);
                    token = strtok(NULL, " ");
                }
                lno++;
            }
		}
	}

	root = reduce_tree(root);
	inorder(root);
	return 0;
}


