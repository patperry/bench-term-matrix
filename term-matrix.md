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
-   remove puncuation
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

    # Note: we use a development versions of text2vec
    #
    # Not run:
    # install.packages(c("corpus", "microbenchmark", "quanteda", "tidytext", "tm"))
    # devtools::install_github("dselivanov/text2vec@5a778fb517082c4a4a69f84dd5e9d045a18bc0bf")

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
        f <- corpus::token_filter(stemmer = "english", drop_punct = TRUE,
                                  drop_number = TRUE, drop = stop_words)
        stats <- corpus::term_counts(text, f, ngrams = ngrams,
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
        x <- quanteda:::dfm(text, stem = TRUE, remove_punct = TRUE,
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

    No features removed.

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
    quanteda    .     1      1     1       1           .             1            1             1
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
                corpus = matrix_corpus(text, bigrams = FALSE),
                # quanteda = matrix_quanteda(text, bigrams = FALSE),
                text2vec = matrix_text2vec(text, bigrams = FALSE),
                tidytext = matrix_tidytext(text, bigrams = FALSE),
                tm = matrix_tm(text, bigrams = FALSE),
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
         expr       min        lq     mean    median        uq      max neval
        basic  89.53498  95.82908 106.5431  96.71177  99.34882 151.2908     5
       corpus  68.03241  68.98661  90.2887  76.14655  81.51476 156.7631     5
     quanteda 154.06562 164.14880 179.1695 169.24797 180.05291 228.3322     5
     text2vec 128.76574 130.97956 134.9654 137.79620 138.53920 138.7465     5
     tidytext 162.37039 162.89691 185.8774 179.47793 192.76505 231.8766     5
           tm 676.52099 713.08554 714.3480 720.94074 724.81737 736.3752     5

### Unigrams (reviews)

    bench2 <- make_bench("Unigram, Reviews", text_reviews, bigrams = FALSE)
    plot_bench(bench2)

![](term-matrix_files/figure-markdown_strict/unnamed-chunk-16-1.png)

    print(bench2$results)

    Unit: milliseconds
         expr        min         lq       mean     median        uq        max neval
        basic  1086.5726  1279.9442  1273.5961  1303.7451  1310.902  1386.8163     5
       corpus   699.6055   720.2039   744.0902   750.8419   769.842   779.9578     5
     quanteda  2635.6067  2743.9400  2805.4253  2819.5896  2864.675  2963.3148     5
     text2vec  1522.0380  1571.7102  1608.8052  1580.9192  1622.431  1746.9278     5
     tidytext  2840.4053  2890.6245  2968.3323  2987.8290  2992.953  3129.8498     5
           tm 10519.6174 10988.0952 11073.4739 11198.7255 11326.692 11334.2390     5

### Bigrams (novel)

    bench3 <- make_bench("Bigram, Novel", text_novel, bigrams = TRUE)
    plot_bench(bench3)

![](term-matrix_files/figure-markdown_strict/unnamed-chunk-17-1.png)

    print(bench3$results)

    Unit: milliseconds
         expr       min        lq      mean    median       uq       max neval
       corpus  67.28631  69.74695  75.41903  74.39687  78.4762  87.18883     5
     text2vec 130.67648 131.97450 145.33998 132.93969 144.8866 186.22266     5
     tidytext 190.65293 194.74266 197.20206 195.37390 198.9658 206.27505     5
           tm 691.86198 703.20665 757.04183 704.98177 788.8686 896.29020     5

### Bigrams (reviews)

    bench4 <- make_bench("Bigram, Reviews", text_reviews, bigrams = TRUE)
    plot_bench(bench4)

![](term-matrix_files/figure-markdown_strict/unnamed-chunk-18-1.png)

    print(bench4$results)

    Unit: milliseconds
         expr        min         lq      mean     median         uq        max neval
       corpus   687.5056   746.9445   784.191   785.6219   819.1792   881.7039     5
     text2vec  1572.0105  1574.1451  1652.493  1632.7921  1735.5433  1747.9762     5
     tidytext  2798.0161  2865.7393  2901.237  2915.9908  2932.8095  2993.6278     5
           tm 10636.4576 10799.5699 10874.963 10892.1622 10894.3806 11152.2446     5

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

Of course, there's more to text minining than just term matrices, so if
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
    [1] stringr_1.2.0     magrittr_1.5      dplyr_0.5.0       Matrix_1.2-9      quanteda_0.9.9-65 ggplot2_2.2.1    

    loaded via a namespace (and not attached):
     [1] NLP_0.1-10             Rcpp_0.12.11           compiler_3.4.0         plyr_1.8.4             tokenizers_0.1.4      
     [6] iterators_1.0.8        tools_3.4.0            digest_0.6.12          nlme_3.1-131           evaluate_0.10         
    [11] tibble_1.3.0           gtable_0.2.0           lattice_0.20-35        psych_1.7.5            fastmatch_1.1-0       
    [16] foreach_1.4.3          DBI_0.6-1              microbenchmark_1.4-2.1 parallel_3.4.0         janeaustenr_0.1.4     
    [21] knitr_1.16             text2vec_0.5.0         rprojroot_1.2          grid_3.4.0             data.table_1.10.4     
    [26] R6_2.2.1               foreign_0.8-67         rmarkdown_1.5          purrr_0.2.2.2          tidyr_0.6.3           
    [31] reshape2_1.4.2         backports_1.0.5        scales_0.4.1           codetools_0.2-15       SnowballC_0.5.1       
    [36] htmltools_0.3.6        corpus_0.7.0           mnormt_1.5-5           assertthat_0.2.0       tidytext_0.1.2        
    [41] colorspace_1.3-2       labeling_0.3           stringi_1.1.5          RcppParallel_4.3.20    lazyeval_0.2.0        
    [46] munsell_0.4.3          slam_0.1-40            tm_0.7-1               broom_0.4.2
