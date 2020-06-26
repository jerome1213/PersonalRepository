package sort.sort;

import sort.common.Constant;

import java.util.ArrayList;
import java.util.Random;

public class SortMain {
    protected int numRandomInteger = 10000;

    protected long beforeTime = 0;
    protected long afterTime = 0;
    protected long secDiffTime = 0;

    protected ArrayList<Integer> testRandomList = Constant.testRandomList;

    public SortMain() {
        Random random = new Random();

        if (testRandomList == null) {
            testRandomList = new ArrayList<>();
            for (int i = 0; i < numRandomInteger; i++) {
                testRandomList.add(random.nextInt(numRandomInteger));
            }
            Constant.testRandomList = testRandomList;

            System.out.println("===== Created Random Integer Array =====");
            System.out.println(testRandomList);
            System.out.println("========================================\n");
        }
    }

    public void printDiffTime() {
        System.out.println(testRandomList);

        afterTime = System.currentTimeMillis();
        secDiffTime = (afterTime - beforeTime);
        System.out.println("소요시간 : " + secDiffTime + "ms\n");
    }

    public void swap(int a, int b) {
        int temp = testRandomList.get(a);
        testRandomList.set(a, testRandomList.get(b));
        testRandomList.set(b, temp);
    }
}
