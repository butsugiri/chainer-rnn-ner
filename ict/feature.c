#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#define LINE 512


struct Feature
{
    char ne[LINE], word[LINE], pos[LINE], chunk[LINE];
    int is_upper;
};

struct T
{
    int type, offset;
};

struct Template
{
    struct T *seq;
    int num;
};

enum FeatureType
{
    Y, W, POS, CHK, IU,
    FEATURE_TYPE_NUM
};

char feat_name[][LINE] = {"y", "w", "pos", "chk", "iu"};


#define MAX_SPLIT_NUM 20
void split(char out[MAX_SPLIT_NUM][LINE], int *token_num,
           const char *str, const char *separator)
{
    char str_cp[LINE], *token;
    int idx=0;
    
    strcpy(str_cp, str);
    token = strtok(str_cp, separator);
    if(token != NULL)
    {
        strcpy(out[idx++], token);
        while(token != NULL)
        {
            token = strtok(NULL, separator);
            if(token != NULL)
                strcpy(out[idx++], token);
        }
    }

    *token_num = idx;
}


void escape(char *src, int size)
{
    int i;
    char buf[LINE];
    for(i=0; i<size; i++)
    {
        if(src[i] == '\0')
            break;
        if(src[i] == ':')
        {
            strcpy(buf, src+i+1);
            strcpy(src+i, "__COLON__");
            strcat(src, buf);
        }
    }
}


int makeFeature(const char* line, struct Feature* feat)
{
    static char split_line[MAX_SPLIT_NUM][LINE];
    static int INPUT_TOKEN_NUM = 4;
    int token_num;
    
    split(split_line, &token_num, line, " \t\n");
    if(token_num == INPUT_TOKEN_NUM)
    {
        strcpy(feat->ne,    split_line[0]);
        strcpy(feat->word,  split_line[1]);
        strcpy(feat->pos,   split_line[2]);
        strcpy(feat->chunk, split_line[3]);
        feat->is_upper = isupper(feat->word[0]) ? 1 : 0;
        return 1;
    }
    else
        return 0; // false
}


void getFeatureMember(char *line, const struct Feature *feat, const int type)
{
    switch(type)
    {
    case Y:
        sprintf(line, feat->word);
        break;
    case W:
        sprintf(line, feat->word);
        break;
    case POS:
        sprintf(line, feat->pos);
        break;
    case CHK:
        sprintf(line, feat->chunk);
        break;
    case IU:
        if(feat->is_upper)
            sprintf(line, "True");
        else
            sprintf(line, "False");
        break;
    }
}


int output(const struct Feature *seq, const int seq_size, const int idx,
           const struct Template *template)
{
    int i, type, offset, position;
    char buf[LINE], line[LINE];

    sprintf(line, "\t");
    for(i=0; i<template->num; i++)
    {
        type = template->seq[i].type, offset = template->seq[i].offset;
        sprintf(buf, "%s[%d]", feat_name[type], offset);
        if(i < template->num-1)
            strcat(buf, "|");
        strcat(line, buf);
    }
    strcat(line, "=");
    for(i=0; i<template->num; i++)
    {
        type = template->seq[i].type, offset = template->seq[i].offset;
        position = idx + offset;
        if(position < 0 || position >= seq_size)
            return 0; // error
        getFeatureMember(buf, &seq[position], type);
        if(i < template->num-1)
            strcat(buf, "|");
        strcat(line, buf);
    }

    escape(line, LINE);
    printf("%s", line);
    return 1;
}


struct Template* newTemplate(int arg_num, ...)
{
    va_list list;
    int i, type, offset;
    struct Template *out = (struct Template*)malloc(sizeof(struct Template));
    out->seq = (struct T*)malloc(sizeof(struct T)*arg_num);
    out->num = arg_num;

    va_start(list, arg_num);
    for(i=0; i<arg_num; i++)
    {
        type   = va_arg(list, int);
        offset = va_arg(list, int);
        out->seq[i].type   = type;
        out->seq[i].offset = offset;
    }

    return out;
}

void deleteTemplate(struct Template *templates[], int size)
{
    int i;
    for(i=0; i<size; i++)
    {
        free(templates[i]->seq);
        free(templates[i]);
    }
}


int main()
{
#define MAX_FEAT_NUM 500
    char line[LINE];
    int i, j, temp_size;
    int feat_size = 0;
    struct Feature feats[MAX_FEAT_NUM];
    struct Template *templates[] =
        {
            newTemplate(1, W, -2),
            newTemplate(1, W, -1),
            newTemplate(1, W, 0),
            newTemplate(1, W, +1),
            newTemplate(1, W, +2),
            newTemplate(2, W, -2, W, -1),
            newTemplate(2, W, -1, W,  0),
            newTemplate(2, W,  0, W, +1),
            newTemplate(2, W, +1, W, +2),
            newTemplate(1, POS, -2),
            newTemplate(1, POS, -1),
            newTemplate(1, POS, 0),
            newTemplate(1, POS, +1),
            newTemplate(1, POS, +2),
            newTemplate(2, POS, -2, POS, -1),
            newTemplate(2, POS, -1, POS,  0),
            newTemplate(2, POS,  0, POS, +1),
            newTemplate(2, POS, +1, POS, +2),
            newTemplate(1, CHK, -2),
            newTemplate(1, CHK, -1),
            newTemplate(1, CHK,  0),
            newTemplate(1, CHK, +1),
            newTemplate(1, CHK, +2),
            newTemplate(2, CHK, -2, CHK, -1),
            newTemplate(2, CHK, -1, CHK,  0),
            newTemplate(2, CHK,  0, CHK, +1),
            newTemplate(2, CHK, +1, CHK, +2),
            newTemplate(1, IU, -2),
            newTemplate(1, IU, -1),
            newTemplate(1, IU,  0),
            newTemplate(1, IU, +1),
            newTemplate(1, IU, +2),
            newTemplate(2, IU, -2, IU, -1),
            newTemplate(2, IU, -1, IU,  0),
            newTemplate(2, IU,  0, IU, +1),
            newTemplate(2, IU, +1, IU, +2)
        };
    temp_size = sizeof(templates) / sizeof(struct Template*);

    for(i=0; i<temp_size; i++)
    {
        for(j=0; j<templates[i]->num; j++)
            printf("[%d, %d] ",
                   templates[i]->seq[j].type, templates[i]->seq[j].offset);
        printf("\n");
    }
    
    while(gets(line) != NULL)
    {
        if(makeFeature(line, &feats[feat_size++]) == 0)
        {
            feat_size--;
            for(i=0; i<feat_size; i++)
            {
                printf("%s", feats[i].ne);
                for(j=0; j<temp_size; j++)
                    output(feats, feat_size, i, templates[j]);
                printf("\n");
            }
            printf("\n");
            feat_size = 0; // initialize
        }
    }
    deleteTemplate(templates, temp_size);
    return 0;
}
