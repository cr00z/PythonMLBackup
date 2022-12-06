# Exercise 2.2.8

import simplecrypt

with open("test/passwords.txt", "r") as pwds:
    passwords = pwds.readlines()
with open("test/encrypted.bin", "rb") as inp:
    encrypted = inp.read()
for pwd in passwords:
    try:
        pwd = pwd.strip()
        print(f'password: {pwd} - ', end='')
        decrypted = simplecrypt.decrypt(pwd, encrypted)
        print(f'accept:\n{decrypted}')
        break
    except:
        print('decline!')
