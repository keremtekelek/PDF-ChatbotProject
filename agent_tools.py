from langchain_core.tools import tool
from typing import Union


# Tetiklenebilir bir tool olduğunu belirtir.

@tool
def multiply_operation(a: int, b: int) -> int:
    """
    Kullanıcı iki sayının çarpımını sorduğunda veya matematiksel bir çarpma işlemi yapmak istediğinde bu aracı kullan.
    """
    print(f"\n LLM Agent 'Çarpma İşlemi' tool'unu çağırdı. {a} x {b} hesaplanıyor.")
    return a * b

@tool
def addition_operation(a: int, b:int) -> int:
    """
    Kullanıcı iki sayının toplamını sorduğunda veya matematiksel bir toplama işlemi yapmak istediğinde bu aracı kullan.
    """
    print(f"\n LLM Agent 'Toplama İşlemi' tool'unu çağırdı. {a} + {b} hesaplanıyor.")
    return a + b

@tool
def subtract_operation(a: int, b:int) -> int:
    """
    Kullanıcı iki sayının çıkarma işlemini sorduğunda veya matematiksel bir çıkarma işlemi yapmak istediğinde bu aracı kullan.
    """
    print(f"\n LLM Agent 'Çıkarma İşlemi' tool'unu çağırdı. {a} - {b} hesaplanıyor.")
    return a - b

@tool
def division_operation(a: int, b: int) -> Union[float, str]:
    """
    Kullanıcı iki sayının bölümünü sorduğunda veya matematiksel bir bölme işlemi yapmak istediğinde bu aracı kullan.
    """
    print(f"\n LLM Agent 'Bölme İşlemi' tool'unu çağırdı. {a} / {b} hesaplanıyor.")

    if b == 0:
        return "Hata: Bir sayı sıfıra bölünemez!"
    
    return a / b 

# Yapay zekanın kullanabileceği alet çantası
usable_tools = [multiply_operation, addition_operation, subtract_operation, division_operation]

print(" Tool oluşturuldu!")