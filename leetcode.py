import math
from typing import Counter, List, Optional

class ListNode:
    def __init__(self, val=0, nxt=None):
        self.val = val
        self.next = nxt
        
class Solution:
    def numJewelsInStones(self, jewels: str, stones: str) -> int:
        count = 0
        for stone in stones:
            if stone in jewels:
                count += 1
        return count
    
    def uniqueMorseRepresentations(self, words: List[str]) -> int:
        count = 0
        morseCode = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        listMorseTemp = []
        for word in words:
            morseTemp = ''
            for c in word:
                morseTemp = morseTemp + morseCode[alphabet.index(c)]
            if morseTemp not in listMorseTemp:
                listMorseTemp.append(morseTemp)
                count += 1
        # print(listMorseTemp)
        # print(count)
        return count
    
    def decimalToBinary(self, decimalNumber):
        string = ''
        while(decimalNumber > 0):
            string += str(decimalNumber % 2)
            decimalNumber //= 2
            # print(decimalNumber)
        print(string)
        string = ''.join(reversed(string))
        print(string)
        return string

    def hammingDistance(self, x: int, y: int) -> int:
        return bin(x^y)[2:].count("1")    
    
    def findMaxCandies(self, candies: List[int]) -> int:
        max = 0
        for candy in candies:
            if candy > max:
                max = candy
        return max

    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        max = self.findMaxCandies(candies)
        booleanList = []
        for candy in candies:
            if candy + extraCandies >= max:
                booleanList.append(True)
            else: booleanList.append(False)
        return booleanList

    def checkDividingNumber(self, number):
        flag = True
        numberTemp = number
        while(numberTemp > 0):
            digit = numberTemp % 10
            # print(digit)
            # print(numberTemp)
            if digit == 0:
                flag = False
                break
            elif number % digit != 0:
                flag = False
                break
            numberTemp //= 10
        return flag

    def selfDividingNumbers(self, left: int, right: int) -> List[int]:
        res = []
        for i in range(left, right+1):
            if self.checkDividingNumber(i):
                res.append(i)
        return res

    def toGoatLatin(self, sentence: str) -> str:
        print(sentence.split())

    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        rows = len(matrix) - 1 
        cols = len(matrix[0]) - 1
        for row in range(0, rows):
            for col in range(0, cols):
                if matrix[row][col] != matrix[row+1][col+1]:
                    return False
        return True

    def findComplement(self, num: int) -> int:
        com = ''
        while num > 0 :
            if num % 2 == 1:
                com += '0'
            else:
                com += '1'
            num //= 2
        return int(com[::-1],2)

    def reverseString(self, s: str) -> str:
        newString = ''
        newString = s[::-1]
        return newString

    def reverseWords(self, s: str) -> str:
        result = []
        arr = s.split()
        for i in arr:
            result.append(self.reverseString(i))
        return ' '.join(result)
    
    def deleteAt(self, s: str, pos: [int])->str:
        pos = [rs-2*idx for idx, rs in enumerate(pos)]
        for i in pos:
            s = s[:i-1] + s[i+1:]
        return s

    def normalize(self, s: str)->str:
        return s[1:] if s[0] == '#' else s
    
    def backspaceCompare(self, s: str, t: str) -> bool:
        s = self.normalize(s)
        t = self.normalize(t)
        lens1 = [idx for idx, c in enumerate(s) if c == '#']
        lens2 = [idx for idx, c in enumerate(t) if c == '#']
        str1 = self.deleteAt(s, lens1)
        str2 = self.deleteAt(t, lens2)
        str1 = self.normalize(str1)
        str2 = self.normalize(str2)

        return str1 == str2
    
    def findRow(self, word: str, keyboard: List[str]) -> int:
        for index, row in enumerate(keyboard):
            if word in row:
                break
        return index
    
    def findWords(self, words: List[str]) -> List[str]:
        keyboard = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm']
        results = []
        row = 0
        for word in words:
            wordTemp = word.lower()
            flag = 1
            row = self.findRow(wordTemp[0], keyboard)
            for char in wordTemp:
                if char not in keyboard[row]:
                    flag = 0
                    break
            if flag:
                results.append(word)
        print(results)
        return results

    def calPoints(self, operations: List[str]) -> int:
        results = []
        lenArray = len(operations)
        for idx in range(0, lenArray):
            if operations[idx] != 'C' and operations[idx] != 'D' and operations[idx] != '+':
                results.append(int(operations[idx]))
            elif operations[idx] == 'C':
                results.pop()
            elif operations[idx] == 'D':
                results.append(int(results[len(results)-1])*2)
            elif operations[idx] == '+':
                results.append(int(results[-1]) + int(results[-2]))
        return sum(results)
                
    def fizzBuzz(self, n: int) -> List[str]:
        results = []
        for i in range(1, n+1):
            if i % 5 == 0 and i % 3 == 0:
                results.append('FizzBuzz')
            elif i % 3 == 0:
                results.append('Fizz')
            elif i % 5 == 0:
                results.append('Buzz')  
            else: results.append(str(i))
        return results  


    def fatorical(self, n: int) -> int:
        if n == 1:
            return n
        return n * self.fatorical(n-1)

    def distributeCandies(self, candyType: List[int]) -> int:
        numberCandiesNeed = len(candyType) // 2
        typeCandies = len(set(candyType))
        # print(numberCandiesNeed, typeCandies)
        return min(numberCandiesNeed, typeCandies)
    
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        max = 0
        pos = 0
        for idx, value in enumerate(arr):
            if max < value:
                max = value
                pos = idx
        return pos
    
    def transpose(self, matrix: List[List[int]]) -> List[List[int]]:
        results = []
        m = len(matrix)
        n = len(matrix[0])
        temp = []
        for i in range(0, n):
            for j in matrix:
                temp.append(j[i])
            results.append(temp)
            temp = [] 
        return results

    def isPrimeNumber(self, number: int):
        if number == 1:
            return False
        elif number >= 2:
            for i in range(2, number):
                if number % i == 0:
                    return False
        return True
    
    def countPrimeSetBits(self, left: int, right: int) -> int:
        count = 0
        for number in range(left, right+1):
            if self.isPrimeNumber(bin(number).count('1')):
                count += 1
        return count
    
    # def getPositon(self, number: int, arr: List[int]) -> int:
    #     for idx, i in enumerate(arr):
    #         if i == number:
    #             return idx

    # def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
    #     results = []
    #     lenS2 = len(nums2)
    #     for i in nums1:
    #         if self.getPositon(i, nums2)+1 >= lenS2:
    #             results.append(-1)
    #         elif i < nums2[self.getPositon(i, nums2)+1]:
    #             results.append(nums2[self.getPositon(i, nums2)+1])
    #         elif i >= nums2[self.getPositon(i, nums2)+1]:
    #             results.append(-1)
    #     return results

    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        results = []
        lenS2 = len(nums2)
        for i in nums1:
            flag = 0
            for index, j in enumerate(nums2):
                if i == j:
                    if index+1 == lenS2:
                        results.append(-1)
                    else:
                        for k in range(index+1, lenS2):
                            if i < nums2[k]:
                                flag = 1
                                results.append(nums2[k])
                                break
                        if flag == 0:
                            results.append(-1)        
        return results

    def findLUSlength(self, a: str, b: str) -> int:
        print(max(10, 10))
        return 0

    def canWinNim(self, n: int) -> bool:
        return False if n % 4 == 0 else True

    def singleNumber(self, nums: List[int]) -> int:
        newSet = set(nums)
        for i in newSet:
            if nums.count(i) == 1:
                return i

    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        max = 0
        count = 1
        for i in range(0, len(nums)-1):
            if nums[i] == 1 and nums[i+1] == 1:
                count += 1
                if max < count:
                    max = count
            else:
                count = 1
        return max
    

    def uncommonFromSentences(self, s1: str, s2: str) -> List[str]:
        results = []
        newString = s1 + ' ' + s2
        newString = newString.split()
        newSet = set(newString)
        for i in newSet:
            if newString.count(i) < 2:
                results.append(i)
        return results
    
    def detectCapitalUse(self, word: str) -> bool:
        stringCheck = []
        stringCheck.append(word.lower())
        stringCheck.append(word.upper())
        stringCheck.append((word.lower()).capitalize())
        if word in stringCheck:
            return True
        else:
            return False
        # return stringCheck

    def getSumDigit(self, num: int) -> int:
        ans = 0
        while num > 0:
            ans += num % 10
            num //= 10
        return ans

    def addDigits(self, num: int) -> int:
        ans = self.getSumDigit(num)
        while ans >= 10:
            ans = self.getSumDigit(ans)
            # print(ans)
        return ans

    def moveZeroes(self, nums: List[int]) -> None:
        results = nums
        for i in range(0, len(nums)):
            if nums[i] == 0:
                print(i)
                results.pop(i)
                results.append(0)
        return results
    
    def findTheDifference(self, s: str, t: str) -> str:
        for c in t:
            if t.count(c) != s.count(c):
                return c
            
    def minMoves(self, nums: List[int]) -> int:
        pass

    def isMonotonic(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True
        flag1 = False
        flag2 = False
        for i in range(1, len(nums)):
            if nums[i-1] >= nums[i]:
                flag1 = True
            else:
                flag1 = False
                break
        for i in range(1, len(nums)):
            if nums[i-1] <= nums[i]:
                flag2 = True
            else:
                flag2 = False
                break
        if flag1 == True or flag2 == True:
            return True
        else:
            return False
        
    def constructRectangle(self, area: int) -> List[int]:
        # initialize width and length with square root of target area rounded down and up, respectively
        width = int(math.sqrt(area))
        length = int(math.ceil(area / width))
        
        # loop until width and length multiplied is equal to target area
        while width * length != area:
            # if width * length is less than target area, increment length
            if width * length < area:
                length += 1
            # if width * length is greater than target area, decrement width
            else:
                width -= 1
        
        # return [length, width]
        return [length, width] 

    def romanToInt(self, s: str) -> int:
        results = 0
        roman = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }
        lenS = len(s)
        for i in range(0, lenS):
            if i < lenS - 1 and roman[s[i]] < roman[s[i+1]]:
                results -= roman[s[i]]
            else:
                results += roman[s[i]]
        return results       

    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        p1, p2 = 0, 1
        while not (numbers[p1] + numbers[p2] == target):
            if numbers[p1] + numbers[p2] < target:
                p2+=1
                p1+=1
            else:
                p1-=1; 
        return [p1+1, p2+1]

    def sortDescending(self, array: List[int]) -> List[int]:
        arrayTemp = array
        for i in range(0, len(arrayTemp)-1):
            for j in range(i+1, len(arrayTemp)):
                if arrayTemp[i] < arrayTemp[j]:
                    arrayTemp[i], arrayTemp[j] = arrayTemp[j], arrayTemp[i]
        return arrayTemp
        
    def findRelativeRanks(self, score: List[int]) -> List[str]:
        sorted_scores = sorted(score, reverse=True)

        rank_dict = {}

        for i, s in enumerate(sorted_scores):
            if i == 0:
                rank_dict[s] = "Gold Medal"
            elif i == 1:
                rank_dict[s] = "Silver Medal"
            elif i == 2:
                rank_dict[s] = "Bronze Medal"
            else:
                rank_dict[s] = str(i + 1)
        
        print(rank_dict)
        result = []

        for s in score:
            result.append(rank_dict[s])

        return result

    def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
        restaurants = {}
        for i in range(0, len(list1)):
            s = 0
            for j in range(0, len(list2)):
                if list1[i] == list2[j]:
                    s = s + i + j
                    restaurants[list1[i]] = s
        # print(restaurants)
        minimum = min(restaurants.values())
        results = []
        for restaurant in restaurants:
            if restaurants[restaurant] == minimum:
                results.append(restaurant)
        return results
    
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        return (Counter(nums1) & Counter(nums2)).elements()


    def reverseStr(self, s: str, k: int) -> str:
        # Convert the string to a list of characters
        lst = list(s)
        
        # Reverse substrings of length k every 2k characters
        for i in range(0, len(lst), 2*k):
            print(i, i+k)
            lst[i:i+k] = reversed(lst[i:i+k])
        
        # Convert the list back to a string and return it
        return "".join(lst)
    
    def dominantIndex(self, nums: List[int]) -> int:
        newArray = sorted(nums, reverse=True)
        if newArray[0] < 2 * newArray[1]:
            return -1
        else:
            return nums.index(newArray[0])
    
    def longestWord(self, words: List[str]) -> str:
        dictString = {''}
        words.sort()
        res = ''
        for word in words:
            if word[:-1] in dictString:
                dictString.add(word)
                if len(word) > len(res):
                    res = word
        return res
    
    def getSumDigitSquared(self, n: int) -> int:
        s = 0
        while n > 0:
            s += (n%10) ** 2
            n //= 10
        return s

    def isHappy(self, n: int) -> bool:
        s = self.getSumDigitSquared(n)
        i = 0
        flag = False
        while i <= 10:
            i += 1
            print(s)
            if s == 1:
                flag = True
                break
            s = self.getSumDigitSquared(s)
        return flag
            
    def findLHS(self, nums: List[int]) -> int:
        freq = Counter(nums)
        max_length = 0
        print(freq)
        for key in freq:
            print(key + 1)
            if key + 1 in freq:
                max_length = max(max_length, freq[key] + freq[key+1])
                
        return max_length
    
    def reverse(self, x: int) -> int:
        numTemp = x
        x = abs(x)
        res = 0
        while x > 0:
            res += (x % 10)
            res *= 10
            x //= 10
        if numTemp < 0:
            res = - (res//10)
        else:
            res //= 10

        if (res < (-2)**31) or (res > (2)**31 - 1):
            return 0
        else:
            return res
        
    def hammingWeight(self, n: int) -> int:
        count=0
        print(n)
        while n!=0:
            if n%2==1:
                count+=1
            n=n//2
        return count
        
    def searchInsert(self, nums: List[int], target: int) -> int:
        pos = 0
        for i in range(0, len(nums)):
            print(i, nums[i])
            if nums[i] >= target:
                pos = i
                break
            if i == len(nums) - 1:
                return len(nums)
        return pos

    def findErrorNums(self, nums: List[int]) -> List[int]:
        dictNum = Counter(nums)
        # print(dictNum)
        res = []
        for i in dictNum.keys():
            if dictNum[i] == 2:
                res.append(i)
                break
        for i in range(1, len(nums)+1):
            if i not in dictNum.keys():
                res.append(i)
                break
        return 
    
    def plusOne(self, digits: List[int]) -> List[int]:
        temp = ''
        res = []
        for i in digits:
            temp += str(i)
        temp = str(int(temp) + 1)
        for i in temp:
            res.append(int(i))
        return res
    
    
    def reverseVowels(self, s: str) -> str:
        vowels = ['a', 'e', 'i', 'o', 'u']
        middle = len(s) // 2
        p1 = 0
        p2 = len(s) - 1
        s = list(s)
        while p1 <= p2:
            if s[p1].lower() not in vowels:
                p1 += 1
                continue
            elif s[p2].lower() not in vowels:
                p2 -= 1 
                continue

            if s[p1].lower() in vowels and s[p2].lower() in vowels:
                s[p1], s[p2] = s[p2], s[p1]
                p1 += 1
                p2 -= 1
        return ''.join(s)  
    
    def repeatedSubstringPattern(self, s: str) -> bool:
        return s in (s + s)[1:-1]
    
    def pascalTriangle(self, numRows: int) -> List[List[int]]:
        finnalRow = []
        finnalRow.append([1])
        for i in range(numRows-1):
            newRow = [1]
            for j in range(i):
                newRow.append(finnalRow[i][j]+finnalRow[i][j+1])
            newRow.append(1)
            finnalRow.append(newRow)
        return finnalRow
    
    def getRow(self, rowIndex: int) -> List[int]:
        res = self.pascalTriangle(rowIndex+1)
        return res[len(res)-1]
    
    def countSegments(self, s: str) -> int:
        return len(s.split())
    
    def trailingZeroes(self, n: int) -> int:
        # sys.set_int_max_str_digits(100000)
        if n == 0: 
            return 0
        res = 1
        for i in range(2, n+1):
            res *= i

        res = str(res)


        print(res, res[len(res)-1])
        if res[len(res)-1] != '0':
            return 0
        else:
            count = 1
            for i in range(len(res)-1, 0, -1):
                if res[i] == '0' and res[i-1] == '0':
                    count += 1
                else:
                    break

        return count

    def longestPalindrome(self, s: str) -> int:
        dictChar = Counter(s)
        maxlengthEvenChar = 0
        maxlengthOddChar = 0
        for i in dictChar.keys():
            if dictChar[i] % 2 == 0:
                maxlengthEvenChar += dictChar[i]
            else:
                if maxlengthOddChar < dictChar[i]:
                    maxlengthOddChar = dictChar[i]
        return maxlengthEvenChar + maxlengthOddChar
    
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        current = head
        while current is not None:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        return prev

        
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        current = head
        while current is not None:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        return prev

class MyQueue:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.front = []   # initialize front stack
        self.back = []    # initialize back stack

    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        self.back.append(x)    # add element to back stack

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        self.peek()            # move elements from back stack to front stack
        return self.front.pop() # remove and return the first element from front stack

    def peek(self) -> int:
        """
        Get the front element.
        """
        if not self.front:     # if front stack is empty
            while self.back:   # move all elements from back stack to front stack
                self.front.append(self.back.pop())
        return self.front[-1]  # return the last element in front stack (i.e., the first element in queue)

    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return not self.front and not self.back  # queue is empty if both stacks are empty


def main():
    # Solution 1: Jewels and Stones.
    # jewels = "aA"
    # stones = "aAAbbbb"
    # Sol = Solution()
    # solved = Sol.numJewelsInStones(jewels, stones)
    # print(solved)

    # Solution 2: Unique Morse Code Words
    # words = ["gin","zen","gig","msg"]
    # words = ["a"]
    # Sol = Solution()
    # solved = Sol.uniqueMorseRepresentations(words)

    # Solution 3: Hamming Distance.
    # Sol = Solution()
    # solved = Sol.hammingDistance(1, 4)
    # print(solved)
    
    # Solution 4: Kids With the Greatest Number of Candies
    # Sol = Solution()
    # solved = Sol.kidsWithCandies(candies = [2,3,5,1,3], extraCandies = 3)
    # print(solved)

    # Solutuon 5: Self Dividing Numbers
    # Sol = Solution()
    # solved = Sol.selfDividingNumbers(left = 1, right = 22)
    # print(solved)

    # Solution 6: Goat Latin
    # Sol = Solution()
    # solved = Sol.toGoatLatin(sentence = "The quick brown fox jumped over the lazy dog")
    # print(solved)

    # Solution 7: Toeplitz Matrix
    # Sol = Solution()
    # solved = Sol.isToeplitzMatrix(matrix = [[1,2,3,4],[5,1,2,3],[9,5,1,2]])
    # print(solved)

    # Solution 8: Number Complement
    # Sol = Solution()
    # solved = Sol.findComplement(num = 2)
    # print(solved)

    # Solutiom 9: Reverse Words in a String III
    # Sol = Solution()
    # solved = Sol.reverseWords(s = "Let's take LeetCode contest")
    # print(solved)

    # Solution 10: Backspace String Compare
    # Sol = Solution()
    # solved = Sol.backspaceCompare(s = "ab##", t = "c#d#")
    # print(solved)

    # Solution 15: Keyboard Row
    # Sol = Solution()
    # solved = Sol.findWords(["Hello","Alaska","Dad","Peace"])
    # print(solved)
    
    # Solution 16: Baseball Game
    # Sol = Solution()
    # solved = Sol.calPoints(operations = ["5","2","C","D","+"])
    # print(solved)
    
    # Solution 18: Fizz Buzz
    # Sol = Solution()
    # solved = Sol.fizzBuzz(n = 5)
    # print(solved)

    # Solution 19: Distribute Candies
    # Sol = Solution()
    # solved = Sol.distributeCandies(candyType = [1,1,2,2,3,3])
    # print(solved)

    # Solution 21: Peak Index in a Mountain Array
    # Sol = Solution()
    # solved = Sol.peakIndexInMountainArray(arr = [0,10,5,2])
    # print(solved)

    # Solution 23: Transpose Matrix
    # Sol = Solution()
    # solved = Sol.transpose(matrix = [[1,2,3],[4,5,6],[7,8,9]])
    # print(solved)
    
    # Solution 24: Prime Number of Set Bits in Binary Representation
    # Sol = Solution()
    # solved = Sol.countPrimeSetBits(left = 6, right = 10)
    # print(solved)

    # Solution 25: Next Greater Element I
    # Sol = Solution()
    # solved = Sol.nextGreaterElement(nums1 = [4,1,2], nums2 = [1,3,4,2])
    # print(solved)

    # Solution 27: Longest Uncommon Subsequence I
    # Sol = Solution()
    # solved = Sol.findLUSlength(a = "aba", b = "cdc")
    # print(solved)

    
    # Solution 31: Single Number
    # Sol = Solution()
    # solved = Sol.singleNumber(nums = [4,1,2,1,2])
    # print(solved)


    # Solution 33: Max Consecutive Ones
    # Sol = Solution()
    # solved = Sol.findMaxConsecutiveOnes(nums = [1,1,0,1,1,1])
    # print(solved)

    # Solution 34: Uncommon Words from Two Sentences
    # Sol = Solution()
    # solved = Sol.uncommonFromSentences( s1 = "d b zu d t", s2 = "udb zu ap")
    # print(solved)

    # Solution 38: Detect Capital
    # Sol = Solution()
    # solved = Sol.detectCapitalUse(word = "USA")
    # print(solved)

    # Solution 39: Add Digits
    # Sol = Solution()
    # solved = Sol.addDigits(num = 38)
    # print(solved)

    # Solution 42: Move Zeroes
    # Sol = Solution()
    # solved = Sol.moveZeroes(nums = [0, 0, 1])
    # print(solved)

    # Solution 43: Find the Difference
    # Sol = Solution()
    # solved = Sol.findTheDifference(s = "abcd", t = "abcde")
    # print(solved)

    # Solution 49: Monotonic Array
    # Sol = Solution()
    # solved = Sol.isMonotonic(nums = [1,2,2,3])
    # print(solved)

    # Solution 64: Two Sum II - Input Array Is Sorted
    # Sol = Solution()
    # solved = Sol.twoSum(numbers = [2,7,11,15], target = 9)
    # print(solved)

    # Solution 70: Relative Ranks
    # Sol = Solution()
    # solved = Sol.findRelativeRanks(score = [10,3,8,9,4])
    # print(solved)
    
    # Solution 73: Minimum Index Sum of Two Lists
    # Sol = Solution()
    # solved = Sol.findRestaurant(list1 = ["Shogun","Tapioca Express","Burger King","KFC"], 
    #                             list2 = ["KFC","Shogun","Burger King"])
    # print(solved)
    
    # Solutions 82: Intersection of Two Arrays II
    # Sol = Solution()
    # solved = Sol.intersect(nums1 = [1,2,2,1], nums2 = [2,2])
    # print(solved)

    # Solution 87: Solution with step by step explanation
    # Sol = Solution()
    # solved = Sol.reverseStr(s = "abcdefg", k = 2)
    # print(solved)

    # Solution 89: Largest Number At Least Twice of Others
    # Sol = Solution()
    # solved = Sol.dominantIndex(nums = [3,6,1,0])
    # print(solved)

    # Solution 93: Longest Word in Dictionary
    # Sol = Solution()
    # solved = Sol.longestWord(words = ["a","banana","app","appl","ap","apply","apple"])
    # print(solved)

    # Solution 96: Happy Number
    # Sol = Solution()
    # solved = Sol.isHappy(n = 19)
    # print(solved)

    # Solution 97: Longest Harmonious Subsequence
    # Sol = Solution()
    # solved = Sol.findLHS(nums = [1,3,2,2,5,2,3,7])
    # print(solved)

    # Solution 104: Reverse Only Letters
    # Sol = Solution()
    # solved = Sol.reverseOnlyLetters()
    # print(solved)

    # Solution 108: Reverse Integer
    # Sol = Solution()
    # solved = Sol.reverse(x = 1534236469)
    # print(solved)

    # Solution 109:  Number of 1 Bits
    # Sol = Solution()
    # solved = Sol.hammingWeight(n = 00000000000000000000000000001011)
    # print(solved)

    # Solution 116: Set Mismatch
    # Sol = Solution()
    # solved = Sol.findErrorNums([2,2])
    # print(solved)
    
    # Solution 121: Plus One
    # Sol = Solution()
    # solved = Sol.plusOne(digits = [1,2,3])
    # print(solved)
    
    # Solution 122: Reverse Vowels of a String
    # Sol = Solution()
    # solved = Sol.reverseVowels(s = "Marge, let's \"went.\" I await news telegram.")
    # print(solved)
    
    # Solution 126: Repeated Substring Pattern
    # Sol = Solution()
    # solved = Sol.repeatedSubstringPattern("ababba")
    # solved = Sol.repeatedSubstringPattern("abab")
    # print(solved)

    # Solution 118 + 128: Pascal triangle
    # Sol = Solution()
    # solved1 = Sol.pascalTriangle(numRows = 5)
    # print(solved1)
    # solved2 = Sol.getRow(rowIndex=3)
    # print(solved2)

    # Solution 135: Number of Segments in a String
    # Sol = Solution()
    # solved = Sol.countSegments(s = "Hello, my name is John")
    # print(solved)

    # Solution 77: Longest Palindrome
    Sol = Solution()
    solved = Sol.longestPalindrome(s = "abccccdd")
    print(solved)

    pass

if __name__ == "__main__":
    main()