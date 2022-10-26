#ifndef __BTREE_H__
#define  __BTREE_H__

#include <iostream>

struct Item{
    float key;
    float value;
    bool operator==(const Item b) const
    {
        return this->key == b.key;
    }
    bool operator!=(const Item b) const
    {
        return this->key != b.key;
    }
    bool operator<=(const Item b) const
    {
        return this->key <= b.key;
    }
    bool operator<(const Item b) const
    {
        return this->key < b.key;
    }
    bool operator>=(const Item b) const
    {
        return this->key >= b.key;
    }
    bool operator>(const Item b) const
    {
        return this->key > b.key;
    }
};


typedef struct _tagBTreeNode
{
    int key_count;
    Item* key;
    _tagBTreeNode** child;
    bool leaf;
}BTreeNode, *PBTreeNode;

class BTree
{
public:
    BTree(int _t);
    ~BTree();

    bool Search(Item key);
    bool Insert(Item key);
    bool Delete(Item key);
    void Display(){Print(root);};

private:
    BTreeNode* SearchNode(PBTreeNode pNode, Item key);
    PBTreeNode AllocateNode();
    void SplitChild(PBTreeNode pParent, int index, PBTreeNode pChild);
    PBTreeNode UnionChild(PBTreeNode pParent, PBTreeNode pCLeft, PBTreeNode pCRight, int index);
    void InsertNonfull(PBTreeNode pNode, Item key);
    Item Max(PBTreeNode pNode);
    Item Min(PBTreeNode pNode);
    bool DeleteNonHalf(PBTreeNode pNode, Item key);
    void DellocateNode(PBTreeNode pNode);
    void DeleteTree(PBTreeNode pNode);
    void Print(PBTreeNode pNode);
private:
    PBTreeNode root;
    int t;     //btree's degree
};

BTree::BTree(int _t){
    root = NULL;
    t = _t;
};

BTree::~BTree()
{
    DeleteTree(root);
    delete root;
};

PBTreeNode BTree::AllocateNode()
{
    PBTreeNode pTemp = new BTreeNode;
    pTemp->key = new Item[2 * t];
    pTemp->child = new PBTreeNode[2 * t + 1];

    struct Item tempt;
    tempt.key = 0;
    tempt.value = 0;
    for(int i = 0; i < 2 * t ; i++)
    {
        pTemp->key[i] = tempt;
        pTemp->child[i] = NULL;
    }
    pTemp->child[2 * t] = NULL;

    return pTemp;
}

PBTreeNode BTree::SearchNode(PBTreeNode pNode, Item key)
{
    int i = 1;
    while (i <= pNode->key_count && key > pNode->key[i])
        i++;

    if (i <= pNode->key_count && key == pNode->key[i])
        return pNode;

    if (pNode->leaf)
        return NULL;
    else
        return SearchNode(pNode->child[i], key);
}

bool BTree::Insert(Item key)
{
    PBTreeNode r = root;
    if(r == NULL)
    {
        r = AllocateNode();
        r->leaf = true;
        r->key_count = 0;

        root = r;
    }

    if (r !=NULL && r->key_count == (2 * t - 1))
    {
        PBTreeNode s = AllocateNode();
        root = s;
        s->leaf = false;
        s->key_count = 0;
        s->child[1] = r;
        SplitChild(s, 1, r);
        InsertNonfull(s, key);
    }
    else
    {
        InsertNonfull(r, key);
    }

    return true;
}

void BTree::SplitChild(PBTreeNode pParent, int index, PBTreeNode pChild)
{
    PBTreeNode pChildEx = AllocateNode();
    pChildEx->leaf = pChild->leaf;
    pChildEx->key_count = t - 1;

    for(int j = 1; j < t; j++)
        pChildEx->key[j] = pChild->key[j + t];

    if(!pChild->leaf)
        for(int j = 1; j <= t; j++)
            pChildEx->child[j] = pChild->child[j + t];

    pChild->key_count = t - 1;

    for(int j = pParent->key_count + 1; j > index; j--)
        pParent->child[j + 1] = pParent->child[j];
    pParent->child[index + 1] = pChildEx;

    for(int j = pParent->key_count; j >= index; j--)
        pParent->key[j + 1] = pParent->key[j];
    pParent->key[index] = pChild->key[t];
    pParent->key_count++;
}

void BTree::InsertNonfull(PBTreeNode pNode, Item key)
{
    int i = pNode->key_count;

    if(pNode->leaf)
    {
        while(i >= 1 && key < pNode->key[i])
        {
            pNode->key[i + 1] = pNode->key[i];
            i--;
        }

        pNode->key[i + 1] = key;
        pNode->key_count++;
    }
    else
    {
        while(i >= 1 && key < pNode->key[i])
            i--;
        i++;

        if(pNode->child[i]->key_count == (2 * t - 1))
        {
            SplitChild(pNode, i, pNode->child[i]);

            if(key > pNode->key[i])
                i++;
        }

        InsertNonfull(pNode->child[i], key);
    }
}

bool BTree::Delete(Item key)
{
    return DeleteNonHalf(root, key);
}

bool BTree::DeleteNonHalf(PBTreeNode pNode, Item key)
{
    int i = 1;

    while(i <= pNode->key_count && key > pNode->key[i])
        i++;

    if(pNode->leaf)//case 1
    {
        if(i <= pNode->key_count && key == pNode->key[i])
        {
            for(int j = i; j < pNode->key_count; j++)
                pNode->key[j] = pNode->key[j + 1];
            pNode->key_count--;

            return true;
        }
        else
        {
            return false;//没有找到要删除的关键字
        }
    }

    if (i <= pNode->key_count && key == pNode->key[i])//case 2
    {
        if (pNode->child[i]->key_count >= t)//case a
        {
            key = Max(pNode->child[i]);
            pNode->key[i] = key;

            return DeleteNonHalf(pNode->child[i], key);
        }
        else if(pNode->child[i + 1]->key_count >= t)//case b
        {
            key = Min(pNode->child[i + 1]);
            pNode->key[i] = key;

            return DeleteNonHalf(pNode->child[i + 1], key);
        }
        else//case c
        {
            PBTreeNode pChild = UnionChild(pNode, pNode->child[i], pNode->child[i + 1], i);

            return DeleteNonHalf(pChild, key);
        }
    }
    else if(pNode->child[i]->key_count == t - 1)//case 3
    {
        if ( i > 1 && pNode->child[i - 1]->key_count >= t)//a_left
        {
            PBTreeNode pMidNode = pNode->child[i];
            PBTreeNode pPreNode = pNode->child[i - 1];

            int nPreNodeKeyCount = pPreNode->key_count;

            for (int j = pMidNode->key_count + 1; j > 1; j--)
                pMidNode->key[j] = pMidNode->key[j - 1];
            pMidNode->key[1] = pNode->key[i - 1];

            for (int j = pMidNode->key_count + 2; j > 1; j--)
                pMidNode->child[j] = pMidNode->child[j - 1];
            pMidNode->child[1] = pPreNode->child[nPreNodeKeyCount + 1];
            pMidNode->key_count++;

            pNode->key[i - 1] = pPreNode->key[nPreNodeKeyCount];

            struct Item tempt;
            tempt.key = 0;
            tempt.value = 0;
            pPreNode->key[nPreNodeKeyCount] = tempt;
            pPreNode->key[nPreNodeKeyCount + 1] = tempt;
            pPreNode->key_count--;

            return DeleteNonHalf(pMidNode, key);
        }
        else if( i <= pNode->key_count && pNode->child[i + 1]->key_count >= t)//a_right
        {
            PBTreeNode pMidNode = pNode->child[i];
            PBTreeNode pNextNode = pNode->child[i + 1];

            int nNextNodeKeyCount = pNextNode->key_count;
            int nMidNodeKeyCount = pMidNode->key_count;

            pMidNode->key[nMidNodeKeyCount + 1] = pNode->key[i];
            pMidNode->child[nMidNodeKeyCount + 2] = pNextNode->child[1];
            pMidNode->key_count++;

            pNode->key[i] = pNextNode->key[1];

            for( int j = 1; j < nNextNodeKeyCount; j++)
                pNextNode->key[j] = pNextNode->key[j + 1];
            for( int j = 1; j <= nNextNodeKeyCount; j++)
                pNextNode->child[j] = pNextNode->child[j + 1];
            pNextNode->key_count--;

            return DeleteNonHalf(pMidNode, key);
        }
        else//b
        {
            if (i > pNode->key_count)//当i指向最后一个关键字的时候，合并的时候要往前移动一步
                i--;

            PBTreeNode pChild = UnionChild(pNode, pNode->child[i], pNode->child[i + 1], i);

            return DeleteNonHalf(pChild, key);
        }
    }

    return DeleteNonHalf(pNode->child[i], key);
}

PBTreeNode BTree::UnionChild(PBTreeNode pParent, PBTreeNode pCLeft, PBTreeNode pCRight, int index)
{
    for(int i = 1; i < t; i++)
        pCLeft->key[t + i] = pCRight->key[i];
    pCLeft->key[t] = pParent->key[index];

    for(int i = 1; i <= t; i++)
        pCLeft->child[t + i] = pCRight->child[i];
    pCLeft->key_count = 2 * t - 1;

    for(int i = index; i < pParent->key_count; i++)
        pParent->key[i] = pParent->key[i + 1];

    for(int i = index + 1; i <= pParent->key_count; i++)
        pParent->child[i] = pParent->child[i + 1];
    pParent->key_count--;

    DellocateNode(pCRight);

    if (pParent->key_count == 0)
    {
        DellocateNode(root);
        root = pCLeft;
    }

    return pCLeft;
}

void BTree::DellocateNode(PBTreeNode pNode)
{
    delete[] pNode->key;
    delete[] pNode->child;
    delete pNode;
}

Item BTree::Max(PBTreeNode pNode)
{
    while (!pNode->leaf)
    {
        pNode = pNode->child[pNode->key_count + 1];
    }

    return pNode->key[pNode->key_count];
}

Item BTree::Min(PBTreeNode pNode)
{
    while (!pNode->leaf)
    {
        pNode = pNode->child[1];
    }

    return pNode->key[1];
}

void BTree::DeleteTree(PBTreeNode pNode)//最后的根元素没有被删除 但是根里的关键字数组和孩子指针数组已经删除
{
    if(pNode->leaf)
    {
        delete[] pNode->key;
        delete[] pNode->child;
    }
    else
    {
        for(int i = 1; i <= pNode->key_count + 1; i++)
        {
            DeleteTree(pNode->child[i]);
            delete pNode->child[i];
        }

        delete[] pNode->key;
        delete[] pNode->child;
    }
}

void BTree::Print(PBTreeNode pNode)
{
    if(pNode->leaf)
    {
        std::cout << "leaf key_count = " << pNode->key_count << " key list :" ;

        for(int i = 1; i <= pNode->key_count;  i++)
        {
            //std::cout << " i = " << i << std::endl;
            std::cout << pNode->key[i].key << " , ";
        }

        std::cout << std::endl;
    }
    else
    {
        for(int i = 1; i <= pNode->key_count + 1; i++)
            Print(pNode->child[i]);

        std::cout << "inner node key_count " << pNode->key_count << " key list :" ;

        for(int i = 1; i <= pNode->key_count; i++)
        {
            //std::cout << " i = " << i << std::endl;
            std::cout << pNode->key[i].key << " , ";
        }
        std::cout << std::endl;
    }
}

bool BTree::Search(Item key) {
    return SearchNode(root, key) != NULL;
}

#endif // __BTREE_H__