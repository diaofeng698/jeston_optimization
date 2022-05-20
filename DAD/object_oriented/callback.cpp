
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
class AddClass
{
    typedef int (*callbackfun)(int, int, std::shared_ptr<AddClass>);

  private:
    int a, b;

  public:
    static int add(int aa, int bb, std::shared_ptr<AddClass> temp);
    void Calculator(callbackfun pf, int a, int b, std::shared_ptr<AddClass> p);
};

void AddClass::Calculator(callbackfun pf, int a, int b, std::shared_ptr<AddClass> p)
{

    cout << pf(a, b, p) << endl;
}
int AddClass::add(int aa, int bb, std::shared_ptr<AddClass> p)
{

    std::shared_ptr<AddClass> temp = p;
    if (temp)
    {
        temp->a = aa;
        temp->b = bb;
        cout << temp->a << endl;
    }
    cout << "hhh" << endl;
    return temp->a + temp->b;
}
int main()
{
    std::shared_ptr<AddClass> a;
    a = std::make_shared<AddClass>();
    a->Calculator(AddClass::add, 1, 2, a);
    return 0;
}

// No access to no-static member

// typedef int (*callbackfun)(int, int);

// class AddClass
// {
//   private:
//     int a, b;

//   public:
//     static int add(int aa, int bb);
//     static int minus(int aa, int bb);
//     void Calculator(callbackfun pf, int a, int b);
// };
// int AddClass::add(int aa, int bb)
// {

//     return aa + bb;
// }
// int AddClass::minus(int aa, int bb)
// {

//     return aa - bb;
// }

// void AddClass::Calculator(callbackfun pf, int a, int b)
// {
//     cout << pf(a, b) << endl;
// }

// int main()
// {
//     std::shared_ptr<AddClass> a;
//     a->Calculator(AddClass::add, 1, 2);
//     a->Calculator(AddClass::minus, 1, 2);
// }
