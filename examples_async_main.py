"""
示例：在测试/脚本中 main 调用异步方法
"""
import asyncio


async def fetch_data():
    """模拟异步方法"""
    await asyncio.sleep(0.1)
    return "done"


async def run_tests():
    """入口：所有异步逻辑放这里"""
    result = await fetch_data()
    print(result)
    return result


def main():
    """同步 main：用 asyncio.run 执行异步入口"""
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()
