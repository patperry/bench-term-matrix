Term Matrix Comparison
======================

Overview
--------

There are multiple R packages that can transform text data into a matrix
of term frequency counts. This document benchmarks five packages:

-   [corpus](https://github.com/patperry/r-corpus)
-   [quanteda](https://github.com/kbenoit/quanteda)
-   [text2vec](http://text2vec.org/)
-   [tidytext](https://github.com/juliasilge/tidytext)
-   [tm](http://tm.r-forge.r-project.org/)

There are four benchmarks, two for unigrams only, and two for unigrams
and bigrams. In each benchmark, we perform the following sequence of
operations:

-   case fold the text
-   tokenize into words
-   remove punctuation
-   remove numbers
-   remove stop words
-   stem
-   compute bigrams (bigram benchmarks only)
-   compute term frequencies
-   remove terms that appear fewer than five times in the corpus
-   compute a term frequency matrix (text by term)

There are some subtle and not-so-subtle differences in how the five
packages implement these operations, so this is not really an
apples-to-apples comparison, and the outputs are different. Keep that in
mind.

Prelude
-------

We will load the following packages.

    library("Matrix")
    library("dplyr", warn.conflicts = FALSE)
    library("ggplot2")
    library("magrittr")
    library("methods")
    library("stringr")

The remaining packages need to be installed, but we will not load their
namespaces:

    # Note: we use a development versions of corpus, installed
    # by following the instructions at
    # https://github.com/patperry/r-corpus#building-from-source
    #
    # Not run:
    # install.packages(c("microbenchmark", "quanteda", "text2vec", "tidytext", "tm"))

For the first test corpus, we use the 62 chapters from *Pride and
Prejudice*, provided by the
[janeaustenr](https://github.com/juliasilge/janeaustenr) library:

    lines <- (data_frame(text = janeaustenr::prideprejudice)
              %>% mutate(
        linenumber = row_number(),
        chapter = cumsum(str_detect(text, regex("^chapter [\\divxlc]",
                                                ignore_case = TRUE)))))
    text_novel <- c(tapply(lines$text, lines$chapter, paste, collapse = "\n"))

For the second test corpus, we use the 5000 movie reviews provided by
the *text2vec* package:

    text_reviews <- text2vec::movie_review$review
    names(text_reviews) <- text2vec::movie_review$id

We will use the Snowball English stop word list:

    stop_words <- corpus::stopwords("english")

Implementations
---------------

### Basic

As a baseline, we will include a basic implementation, written from
scratch by Dmitriy Selivanov (*text2vec* author) that can handle
unigrams but not bigrams:

    # helper function for normalizing text, also used by text2vec below
    preprocess <- function(x)
    {
        # Note: this works fine for ASCII but not for general Unicode.
        # For Unicode, do the following instead:
        #
        # (stringi::stri_trans_nfkc_casefold(x)
        #  %>% stringi::stri_replace_all_regex("[^\\p{Letter}\\s]", ""))

        str_to_lower(x) %>% str_replace_all("[^[:alpha:]\\s]", "")
    }

    # helper function for tokenizing and stemming, also used by text2vec below
    stem_tokenizer <- function(x)
    {
        str_split(x, boundary("word")) %>% lapply(SnowballC::wordStem, "english")
    }

    matrix_basic <- function(text, min_count = 5)
    {
        # normalize and tokenize the text
        toks <- text %>% preprocess %>% stem_tokenizer
        toks_flat <- unlist(toks, recursive = FALSE, use.names = FALSE)

        # compute the text lengths
        ntok <- vapply(toks, length, 0L)

        # compute the types, remove stop words
        types <- unique(toks_flat) %>% setdiff(stop_words)

        # construct the term matrix
        i <- rep.int(seq_along(text), ntok)
        j <- match(toks_flat, types)
        drop <- is.na(j)
        x <- sparseMatrix(i = i[!drop], j = j[!drop], x = 1,
                          dims = c(length(text), length(types)),
                          dimnames = list(names(text), types),
                          check = FALSE)

        # drop terms below the minimum count
        x <- x[, colSums(x) >= min_count, drop = FALSE]
        x
    }

### corpus

    matrix_corpus <- function(text, bigrams = FALSE, min_count = 5)
    {
        if (bigrams) {
            ngrams <- 1:2
        } else {
            ngrams <- 1
        }
        f <- corpus::text_filter(stemmer = "english", drop_punct = TRUE,
                                 drop_number = TRUE, drop = stop_words)
        stats <- corpus::term_stats(text, f, ngrams = ngrams,
                                    min_count = min_count)
        x <- corpus::term_matrix(text, f, select = stats$term)
        x
    }

### quanteda

    matrix_quanteda <- function(text, bigrams = FALSE, min_count = 5)
    {
        if (bigrams) {
            ngrams <- 1:2
        } else {
            ngrams <- 1
        }
        x <- quanteda::dfm(text, stem = TRUE, remove_punct = TRUE,
                           remove_numbers = TRUE, remove = stop_words,
                           ngrams = ngrams, verbose = FALSE)
        x <- quanteda::dfm_trim(x, min_count = min_count, verbose = FALSE)
        x
    }

### text2vec

    # Written by Dmitriy Selivanov
    matrix_text2vec <- function(text, bigrams = FALSE, min_count = 5)
    {
        if (bigrams) {
            ngram <- c(1, 2)
        } else {
            ngram <- c(1, 1)
        }

        # since we don't care about RAM usage we will tokenize everything only
        # once and do it with a single call to preprocess and tokenizer
        tokens <- preprocess(text) %>% stem_tokenizer
      
        it_train <- text2vec::itoken(tokens, n_chunks = 1, progressbar = FALSE)
        vocab <- text2vec::create_vocabulary(it_train, ngram = ngram,
                                             stopwords = stop_words)
        pruned_vocab <- text2vec::prune_vocabulary(vocab,
                                                   term_count_min = min_count)
        vectorizer <- text2vec::vocab_vectorizer(pruned_vocab)
        x <- text2vec::create_dtm(it_train, vectorizer)
        x
    }

### tidytext

    # Note: this filters punctuation but keeps numbers
    matrix_tidytext <- function(text, bigrams = FALSE, min_count = 5)
    {
        data <- tibble::tibble(text_id = seq_along(text), text = text)
        stops <- tibble::tibble(word = stop_words)

        # unigrams
        freqs <- (data
            %>% tidytext::unnest_tokens(word, text)
            %>% anti_join(stops, by = "word")
            %>% mutate(term = SnowballC::wordStem(word, "english"))
            %>% count(text_id, term)
            %>% ungroup())

        # bigrams
        if  (bigrams) {
            freqs2 <- (data
                %>% tidytext::unnest_tokens(bigram, text, token = "ngrams", n = 2)
                %>% tidyr::separate(bigram, c("type1", "type2"), sep = " ")
                %>% filter(!type1 %in% stop_words,
                           !type2 %in% stop_words)
                %>% mutate(type1 = SnowballC::wordStem(type1, "english"),
                           type2 = SnowballC::wordStem(type2, "english"))
                %>% mutate(term = paste(type1, type2))
                %>% count(text_id, term)
                %>% ungroup())

            freqs <- rbind(freqs, freqs2)
        }

        # form matrix in slam format
        x <- freqs %>% tidytext::cast_dtm(text_id, term, n)

        # remove rare terms
        x <- x[, slam::col_sums(x) >= min_count, drop = FALSE]

        # cast to dgCMatrix format
        x <- sparseMatrix(i = x$i, j = x$j, x = x$v, dims = dim(x),
                          dimnames = dimnames(x), check = FALSE)
        x
    }

### tm

    # from http://tm.r-forge.r-project.org/faq.html#Bigrams
    BigramTokenizer <- function(x)
    {
        unlist(lapply(NLP::ngrams(NLP::words(x), 2), paste, collapse = " "),
               use.names = FALSE)
    }

    matrix_tm <- function(text, bigrams = FALSE, min_count = 5)
    {
        corpus <- (tm::VCorpus(tm::VectorSource(text))
                   %>% tm::tm_map(tm::content_transformer(tolower))
                   %>% tm::tm_map(tm::removeWords, stop_words)
                   %>% tm::tm_map(tm::removePunctuation)
                   %>% tm::tm_map(tm::removeNumbers)
                   %>% tm::tm_map(tm::stemDocument, language = "english"))

        control <- list(wordLengths = c(1, Inf),
                        bounds = list(global = c(min_count, Inf)))

        x <- tm::DocumentTermMatrix(corpus, control = control)

        if (bigrams) {
            control$tokenize <- BigramTokenizer
            x2 <- tm::DocumentTermMatrix(corpus, control = control)

            x <- cbind(x, x2)
        }

        x <- sparseMatrix(i = x$i, j = x$j, x = x$v, dims = dim(x),
                          dimnames = dimnames(x), check = FALSE)
        x
    }

Caveats
-------

These implementations all give different results. See, for example, the
results on the following sample text:

    sample <- "Above ground. Another sentence. Others..."

    # compute term matrices using five implementations
    xs <- list(
        corpus   = matrix_corpus(sample, bigrams = TRUE, min_count = 1),
        quanteda = matrix_quanteda(sample, bigrams = TRUE, min_count = 1),
        text2vec = matrix_text2vec(sample, bigrams = TRUE, min_count = 1),
        tidytext = matrix_tidytext(sample, bigrams = TRUE, min_count = 1),
        tm       = matrix_tm(sample, bigrams = TRUE, min_count = 1))

    # normalize the names (some use '_' to join bigrams, others use ' ')
    for (i in seq_along(xs)) {
        colnames(xs[[i]]) <- str_replace_all(colnames(xs[[i]]), " ", "_")
    }

    # get the unique terms
    terms <- unique(c(sapply(xs, colnames), recursive = TRUE))

    # put unigrams before bigrams, then order lexicographically
    terms <- terms[order(str_count(terms, "_"), terms)]

    # combine everything into a single matrix
    x <- matrix(0, length(xs), length(terms), dimnames = list(names(xs), terms))
    for (i in seq_along(xs)) {
        xi <- xs[[i]]
        x[i, colnames(xi)] <- as.numeric(xi[1, ])
    }

    print(as(x, "dgCMatrix"))

    5 x 9 sparse Matrix of class "dgCMatrix"
             abov anoth ground other sentenc abov_ground anoth_sentenc ground_anoth sentenc_other
    corpus      .     1      1     .       1           .             1            .             .
    quanteda    .     1      1     1       1           1             1            1             1
    text2vec    1     1      1     .       1           1             1            1             .
    tidytext    .     1      1     1       1           .             1            1             1
    tm          .     1      1     1       1           .             1            1             1

    print(sample)

    [1] "Above ground. Another sentence. Others..."

Some major differences between the implementations:

1.  With the *quanteda*, *tidytext*, and *tm* implementations, we remove
    stop words first, and then stem. With *text2vec*, we stem and then
    remove stop words. *Corpus* removes stop words after stemming and by
    default does not stem any words on the drop list. The word "other"
    is a stop word, but "others" is not. However, "others" stems to
    "other". *Corpus* and *text2vec* remove "others"; *quanteda*,
    *tidytext*, and *tm* replace "others" with a non-dropped "other"
    token. Another example: "above" is a stop word that stems to "abov".
    *Text2vec* replaces "above" with "abov" and keeps the token; the
    other packages drop "above".

2.  By design, *corpus* does not form bigrams across dropped tokens, in
    particular across dropped punctuation. The other packagages form
    bigrams from "ground. Another" and "sentence. Others"; corpus does
    not.

There are also differences in how the packages handle numbers and
punctuation. Beyond that, there are differences in the default output
formats, but we have converted everything to the *Matrix* `"dgCMatrix"`
format to make the outputs comparable. (By default, *corpus*,
*quanteda*, and *text2vec* return *Matrix* objects, but *tidytext* and
*tm* return *slam* objects.)

Results
-------

### Setup

First we benchmark the implementations:

    make_bench <- function(name, text, bigrams)
    {
        if (!bigrams) {
            results <- microbenchmark::microbenchmark (
                basic = matrix_basic(text),
                corpus = matrix_corpus(text, bigrams = FALSE),
                quanteda = matrix_quanteda(text, bigrams = FALSE),
                text2vec = matrix_text2vec(text, bigrams = FALSE),
                tidytext = matrix_tidytext(text, bigrams = FALSE),
                tm = matrix_tm(text, bigrams = FALSE),
                times = 5)
        } else {
            results <- microbenchmark::microbenchmark (
                corpus = matrix_corpus(text, bigrams = TRUE),
                quanteda = matrix_quanteda(text, bigrams = TRUE),
                text2vec = matrix_text2vec(text, bigrams = TRUE),
                tidytext = matrix_tidytext(text, bigrams = TRUE),
                tm = matrix_tm(text, bigrams = TRUE),
                times = 5)
        }

        list(name = name, results = results)
    }

    plot_bench <- function(bench, title)
    {
        (ggplot(summary(bench$results),
                aes(x = expr, fill = expr, y = median, ymin = lq, ymax = uq))
         + geom_bar(color = "white", stat = "identity")
         + geom_errorbar()
         + scale_fill_discrete(name = "Implementation")
         + xlab("")
         + ylab("Computation time (less is better)"))
    }

Next, we present the results for the four benchmarks.

### Unigrams (novel)

    bench1 <- make_bench("Unigram, Novel", text_novel, bigrams = FALSE)
    plot_bench(bench1)

![](term-matrix_files/figure-markdown_strict/unnamed-chunk-15-1.png)

    print(bench1$results)

    Unit: milliseconds
         expr       min        lq      mean   median        uq       max neval
        basic  89.95256  92.14857 105.82787 105.4601 118.35586 123.22230     5
       corpus  67.88736  68.00299  72.71438  68.2012  76.66693  82.81344     5
     quanteda 169.15080 173.34062 179.43663 177.6169 179.69622 197.37859     5
     text2vec 125.17750 127.44258 132.88344 128.5781 130.04000 153.17898     5
     tidytext 173.78775 176.47121 186.89969 178.1253 197.31312 208.80111     5
           tm 684.84074 687.54222 691.65563 687.6326 692.39192 705.87074     5

### Unigrams (reviews)

    bench2 <- make_bench("Unigram, Reviews", text_reviews, bigrams = FALSE)
    plot_bench(bench2)

![](term-matrix_files/figure-markdown_strict/unnamed-chunk-16-1.png)

    print(bench2$results)

    Unit: milliseconds
         expr        min         lq       mean     median         uq        max neval
        basic  1041.9600  1260.0894  1310.9323  1269.2078  1415.8655  1567.5389     5
       corpus   728.9617   734.2605   825.1777   780.5213   932.7459   949.3989     5
     quanteda  2721.0763  2808.6665  2877.4857  2842.2347  2989.4144  3026.0365     5
     text2vec  1542.2544  1555.8159  1683.9892  1565.6625  1603.5334  2152.6797     5
     tidytext  3008.3909  3037.8763  3157.5544  3117.0764  3170.6102  3453.8183     5
           tm 10787.6874 10875.7651 11817.5059 10993.2838 13136.4888 13294.3043     5

### Bigrams (novel)

    bench3 <- make_bench("Bigram, Novel", text_novel, bigrams = TRUE)
    plot_bench(bench3)

![](term-matrix_files/figure-markdown_strict/unnamed-chunk-17-1.png)

    print(bench3$results)

    Unit: milliseconds
         expr        min         lq       mean     median         uq        max neval
       corpus   78.47289   78.50051   80.92065   82.51431   82.55469   82.56085     5
     quanteda 2017.87454 2038.97487 2157.33919 2199.38437 2222.40093 2308.06125     5
     text2vec  202.36722  204.97548  213.29986  205.29699  207.20550  246.65413     5
     tidytext  641.90621  681.32893  682.90659  689.73203  698.40658  703.15921     5
           tm 1329.63315 1337.25640 1363.83871 1359.52149 1377.54832 1415.23422     5

### Bigrams (reviews)

    bench4 <- make_bench("Bigram, Reviews", text_reviews, bigrams = TRUE)
    plot_bench(bench4)

![](term-matrix_files/figure-markdown_strict/unnamed-chunk-18-1.png)

    print(bench4$results)

    Unit: seconds
         expr       min        lq      mean    median        uq       max neval
       corpus  1.016665  1.088087  1.136768  1.111102  1.207830  1.260154     5
     quanteda 23.579375 23.780355 24.756739 24.694733 25.161024 26.568208     5
     text2vec  2.686002  2.755716  3.050985  2.846913  3.017942  3.948349     5
     tidytext 12.528027 12.581825 12.926033 12.837822 13.308036 13.374454     5
           tm 21.080682 21.931901 22.073406 22.208548 22.461196 22.684704     5

Summary
-------

*Corpus* is faster than the other packages, by at least a factor of 2
and as much as a factor of 10. What's going on here? The other packages
tokenize the text into a list of character vectors, then the process the
token lists to form the term matrices. *Corpus* instead bypasses the
intermediate step, going directly from the text to the term matrix
without constructing an intermediate "tokens" object. This is only
possible because all of the *corpus* normalization and tokenization is
written directly in C.

The downside of the *corpus* approach is flexibility: if you're using
*corpus*, you can't swap out the normalization or tokenizer for
something custom. With varying degrees of ease, the other packages let
you swap out these steps for your own custom functions.

Of course, there's more to text mining than just term matrices, so if
you need more, than *corpus* alone probably won't be sufficient for you.
The other packages have different strengths: *quanteda* and *text2vec*
provide a host of models and metrics; *tidytext* fits in well with "tidy
data" pipelines built on *dplyr* and related tools; *tm* has lots of
extension packages for data input and modeling. Choose the package that
best needs your needs.

Session information
-------------------

    sessionInfo()

    R version 3.4.0 (2017-04-21)
    Platform: x86_64-apple-darwin16.5.0 (64-bit)
    Running under: macOS Sierra 10.12.5

    Matrix products: default
    BLAS: /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/libBLAS.dylib
    LAPACK: /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/libLAPACK.dylib

    locale:
    [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8

    attached base packages:
    [1] methods   stats     graphics  grDevices utils     datasets  base     

    other attached packages:
    [1] bindrcpp_0.2  stringr_1.2.0 magrittr_1.5  dplyr_0.7.2   Matrix_1.2-9  quanteda_0.99 ggplot2_2.2.1

    loaded via a namespace (and not attached):
     [1] slam_0.1-40            NLP_0.1-10             reshape2_1.4.2         purrr_0.2.2.2          lattice_0.20-35       
     [6] colorspace_1.3-2       htmltools_0.3.6        SnowballC_0.5.1        tidytext_0.1.3         rlang_0.1.1           
    [11] foreign_0.8-67         glue_1.1.1             lambda.r_1.1.9         text2vec_0.5.0         foreach_1.4.3         
    [16] bindr_0.1              plyr_1.8.4             munsell_0.4.3          gtable_0.2.0           futile.logger_1.4.3   
    [21] codetools_0.2-15       psych_1.7.5            evaluate_0.10          labeling_0.3           knitr_1.16            
    [26] tm_0.7-1               parallel_3.4.0         broom_0.4.2            tokenizers_0.1.4       Rcpp_0.12.12          
    [31] scales_0.4.1           backports_1.0.5        corpus_0.9.1.9000      RcppParallel_4.3.20    microbenchmark_1.4-2.1
    [36] fastmatch_1.1-0        mnormt_1.5-5           digest_0.6.12          stringi_1.1.5          grid_3.4.0            
    [41] rprojroot_1.2          tools_3.4.0            lazyeval_0.2.0         tibble_1.3.3           janeaustenr_0.1.4     
    [46] futile.options_1.0.0   tidyr_0.6.3            pkgconfig_2.0.1        data.table_1.10.4      lubridate_1.6.0       
    [51] assertthat_0.2.0       rmarkdown_1.5          iterators_1.0.8        R6_2.2.2               nlme_3.1-131          
    [56] compiler_3.4.0
