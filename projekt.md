Wymagania
Zaliczenie na podstawie wykonania dwóch projektów mających zbadać

możliwości rozwiązania problemu CVRP z użyciem algorytmu mrówkowego (realizowany indywidualnie),
możliwości wykorzystania i usprawnienia podejścia MCTS do implementacji sztucznej inteligencji w wybranej grze (realizowany w parach).
Celem projektu jest przeprowadzenie świadomego i celowego procesu badawczego oraz opisanie jego wyników w sposób przystępny dla innych badaczy. W związku z tym, poza implementacją samych metod na projekcie wymagamy następujących elementów:

1. konspektu – wstępny plan badań, zawierający:

opis problemu,
wstępny przegląd literatury,
hipotezy badawcze, które zostaną zweryfikowane w ramach projektu,
sposób weryfikacji hipotez (w szczególności wykorzystane zbiory danych, sposób przeprowadzenia eksperymentów, plan na opracowanie wyników),
harmonogram działań,
ogólny projekt techniczny – planowane wykorzystanie istniejących rozwiązań i bibliotek (bardzo zgrubne).
Stawiając i weryfikując hipotezy należy pamiętać, że metody bazujące na generatorach liczb losowych mogą dawać różne wyniki w zależności od startowego ziarna. Należy zapewnić wielokrotne powtórzenia eksperymentu z różnym ziarnem losowości, aby stwierdzić czy wyniki są regularnie powtarzalne i jak duży jest ich rozrzut. Program należy zaprojektować w taki sposób, aby możliwe było powtórzenie konkretnego przebiegu z zadanego ziarna generatora.

2. raportu końcowego, zawierającego:
opis problemu (uzupełniony z konspektu),
sprawdzone hipotezy i opis eksperymentów, które posłużyły do ich weryfikacji (rozszerzona wersja z konspektu)
wyniki eksperymentów przedstawione w sposób przystępny dla czytelnika – nie w formie surowych tabel z pojedynczymi wartościami,
wniosków, w tym stwierdzenia czy postawione hipotezy są prawdziwe,
raport powinien zawierać odnośniki bibliograficzne do literatury (recenzowanych publikacji),
streszczenie na około 250 słów podsumowujące eksperymenty i najważniejsze wyniki,
szczegółowa informacja w jakim stopniu i do czego zostały użyte duże modele językowe (LLMy) - zarówno w przypadku kodu jak i sprawozdania.
3. prezentacji wyników przed całą grupą. Prezentacja powinna trwać około 10 minut i prezentować to co zostało zrobione w projekcie + najciekawsze wyniki.

4. kodu źródłowego w postaci pliku zip lub linku do publicznego repozytorium kodu.

Z racji tego, że projekt jest badawczy, a nie inżynierski, program ma być tylko środkiem do przeprowadzenia badania, a nie jego centrum. Najważniejszym elementem oceny są konspekt i raport. Jeśli program będzie miał funkcjonalności, które nie posłużyły do przeprowadzenia badań lub nie zostały uwzględnione w raporcie, nie będą one brane pod uwagę przy ocenie. Architektura programu również nie będzie oceniana. Najważniejszym elementem oceny będzie treść i forma raportu.

Każdy z projektów będzie oceniany do 30 punktów. 5 punktów za konspekt, z naciskiem na przegląd literatury oraz postawione hipotezy i sposób ich weryfikacji. 20 punktów za raport z naciskiem na dobrze przeprowadzone i opracowane eksperymenty i wnioski. 5 punktów za prezentację.

Opóźnienia
-5 pkt za każdy rozpoczęty tydzień opóźnienia wysłania raportu
-3 pkt za nieusprawiedliwioną nieobecności na obowiązkowych zajęciach
-2 pkt za każdy rozpoczęty tydzień opóźnienia prezentacji lub oddania konspektu

Projekt 2
Zastosowanie Monte Carlo Tree Search/Upper Confidence Bound Applied to Trees do stworzenia sztucznej inteligencji grającej w wybraną grę dla dwóch graczy z pełną (doskonałą) informacją.

Wybór gry musi być zaakceptowany przez prowadzącego przez rozpoczęciem prac nad konspektem (w celu zapewnienia, że nie będzie to gra zbyt prosta/trudna dla algorytmu MCTS). Każda grupa powinna wybrać inną grę.

Klasyczne zasady gry można modyfikować na potrzeby eksperymentów, np. zmniejszać/zwiększać wielkość planszy, wprowadzić ograniczenie na maksymalną liczbą ruchów (po której np. następuje remis), pomijać/dodawać jakieś zasady. Wszystkie te modyfikacje powinny być jasno opisane w konspekcie lub ewentualnie w raporcie końcowym z uzasadnieniem (jeśli wyniknęły z jakichś problemów w trakcie przeprowadzania eksperymentów).

W ramach hipotez badawczych uwzględnić porównanie skuteczności graczy:

opartego o UCT bez dodatkowych modyfikacji,
dwóch wariantów usprawnień UCT, przynajmniej jednego zaczerpniętego z literatury,
wybranego podejścia heurystycznego wyspecjalizowanego w danej grze.
Należy także postawić własną hipotezę istotnie inną od powyższych zagadnień - najlepiej związaną z zasadami wybranej gry.

Ponadto należy dostarczyć interfejs, który umożliwi przeprowadzenie rozgrywki człowiek-komputer oraz wykonać i zaraportować takie testy na puli co najmniej 5 różnych osób.

NASZ TEMAT PROJEKTU 2:
Warcaby 10x10 bez obowiązkowego bicia, damki biją po całej diagonali
