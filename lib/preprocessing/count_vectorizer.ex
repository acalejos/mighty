defmodule Mighty.Preprocessing.CountVectorizer do
  @moduledoc ~S"""
  A `CountVectorizer` contains performs a `transform/2` function that converts a corpus into a
  term frequency (TF) matrix and a document frequency (DF) matrix.

  The term frequency matrix is a matrix of shape $\lvert$ corpus $\rvert \times \lvert$ vocabulary $\rvert$ where each row
  is a document and each column is a feature. It represents the count of each feature in each document.
  The document frequency matrix is a vector of length $\lvert$ vocabulary $\rvert$ where each element is the
  number of documents that contain the feature at the corresponding index in the vocabulary divided by the
  number of documents in the corpus (hence the term 'frequency').

  ## Creating a `CountVectorizer`

  There are a number of different options that can be used to control the vocabulary that is built from the corpus
  as well as control the features that are included in the term frequency matrix. You can provide your own vocabulary
  or you can let the `CountVectorizer` build the vocabulary from the corpus. If you provide your own vocabulary, you
  must pass the vocabulary as a list, MapSet, or Map. If you pass a list or MapSet, they must contain strings that
  represent the desired tokens in your vocabulary. If you provide a Map, the keys must be strings that represent the
  desired tokens in your vocabulary and the values must be integers that represent the index of the token in the
  vocabulary. The indices must be consecutive integers from 0..length(vocabulary). If you do not provide your own vocabulary,
  the `CountVectorizer` will build the vocabulary from the corpus using the `preprocessor` and `tokenizer` options. The
  `preprocessor` option is a function that takes a string and returns a string. The `tokenizer` option is a function
  that takes a string and returns a list of strings. The `preprocessor` function is applied to each document in the corpus
  before the `tokenizer` function is applied to each document in the corpus. The `tokenizer` function is applied to each
  document in the corpus before the ngrams are generated. The `ngram_range` option controls the ngrams that are generated
  from the tokens. The `ngram_range` option must be a tuple of integers where the first integer is the minimum ngram length
  and the second integer is the maximum ngram length. All ngrams between your max and min ngram range will be generated,
  meaning `ngram_range: {1,1}` will generate only unigrams, `ngram_range: {2,2}` will only generate bigrams, and
  `ngram_range: {1,2}` will generate both unigrams and bigrams. The `stop_words` option is a list of strings that represent tokens
  that should be removed from the corpus before the vocabulary is built. The `binary` option controls whether the term frequency
  matrix is binary or not. If `binary` is `true`, all non-zero counts are set to 1. This is useful for discrete probabilistic
  models that model binary events rather than integer counts.

  Control the number of features that are included in the term frequency matrix by setting the `max_features` option.
  Control the minimum and maximum document frequency of the features that are included in the term frequency matrix by setting
  the `min_df` and `max_df` options. Control the ngram range of the features that are included in the term frequency
  matrix by setting the `ngram_range` option. Control whether the term frequency matrix is binary or not
  by setting the `binary` option.

  ## Examples

  ```elixir
  iex> alias Mighty.Preprocessing.CountVectorizer, as: CV
  iex> corpus = [
  ...>   "This is the first document",
  ...>   "This document is the second document",
  ...>   "And this is the third one",
  ...>   "Is this the first document"
  ...> ]
  iex> {_,tf} = CV.new() |> CV.fit_transform(corpus)
  iex> tf |> Nx.to_list()
  [
    [0, 1, 1, 1, 0, 0, 1, 0, 1],
    [0, 2, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 1, 1, 1],
    [0, 1, 1, 1, 0, 0, 1, 0, 1]
  ]
  iex> {_,tf} = CV.new(ngram_range: {2, 2}) |> CV.fit_transform(corpus)
  iex> tf |> Nx.to_list()
  [
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]
  ]
  iex> {vec, tf} = CV.new(max_features: 5, ngram_range: {1, 2}, min_df: 2, max_df: 0.8) |> CV.fit_transform(corpus)
  iex> tf |> Nx.to_list()
  [
    [1, 1, 1, 1, 1],
    [2, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 1, 0, 1]
  ]
  iex> vec.vocabulary
  %{
   "document" => 0,
   "first" => 1,
   "first document" => 2,
   "is the" => 3,
   "the first" => 4
  }
  iex> vec.pruned
  MapSet.new(["and", "and this", "document is", "is", "is this", "one", "second",
  "second document", "the", "the second", "the third", "third", "third one",
  "this", "this document", "this is", "this the"])
  ```
  """
  alias Mighty.Utils

  defstruct vocabulary: nil,
            ngram_range: {1, 1},
            max_features: nil,
            min_df: 1,
            max_df: 1.0,
            stop_words: MapSet.new(),
            binary: false,
            preprocessor: nil,
            tokenizer: nil,
            pruned: nil

  defp do_process(vectorizer = %__MODULE__{}, doc) do
    {pre_mod, pre_func, pre_args} = vectorizer.preprocessor
    {token_mod, token_func, token_args} = vectorizer.tokenizer

    doc
    |> then(fn doc -> apply(pre_mod, pre_func, [doc | pre_args]) end)
    |> then(fn doc -> apply(token_mod, token_func, [doc | token_args]) end)
    |> Utils.ngram_range(vectorizer.ngram_range)
    |> Enum.filter(fn token ->
      !(vectorizer.stop_words && token in vectorizer.stop_words)
    end)
  end

  defp build_vocab(vectorizer, corpus) do
    vocabulary =
      case vectorizer.vocabulary do
        nil ->
          corpus
          |> Enum.reduce(MapSet.new(), fn doc, vocab ->
            MapSet.union(vocab, MapSet.new(do_process(vectorizer, doc)))
          end)
          |> Enum.sort()
          |> Enum.with_index()
          |> Enum.into(%{})

        _ ->
          vectorizer.vocabulary
      end

    struct(vectorizer, vocabulary: vocabulary)
  end

  @doc """
  Creates a new CountVectorizer with the given options.

  Returns the new CountVectorizer.
  """
  def new(opts \\ []) do
    opts = Mighty.Preprocessing.Shared.validate_shared!(opts)
    struct(__MODULE__, opts)
  end

  defp where_columns(condition = %Nx.Tensor{shape: {_cond_len}}) do
    count = Nx.sum(condition) |> Nx.to_number()
    Nx.argsort(condition, direction: :desc) |> Nx.slice_along_axis(0, count, axis: 0)
  end

  defp limit_features(
         vectorizer = %__MODULE__{},
         tf = %Nx.Tensor{},
         df = %Nx.Tensor{shape: {df_len}},
         high,
         low,
         limit
       ) do
    mask = Nx.broadcast(1, {df_len})

    mask = if high, do: Nx.logical_and(mask, Nx.less_equal(df, high)), else: mask

    mask = if low, do: Nx.logical_and(mask, Nx.greater_equal(df, low)), else: mask

    limit =
      case limit do
        0 ->
          limit

        nil ->
          limit

        _ ->
          limit - 1
      end

    mask =
      if limit && Nx.to_number(Nx.greater(Nx.sum(mask), limit)) == 1 do
        tfs = Nx.sum(tf, axes: [0]) |> Nx.flatten()
        orig_mask_inds = where_columns(mask)
        mask_inds = Nx.argsort(Nx.take(tfs, orig_mask_inds) |> Nx.multiply(-1))[0..limit]
        new_mask = Nx.broadcast(0, {df_len})
        new_indices = Nx.take(orig_mask_inds, mask_inds) |> Nx.new_axis(1)
        new_updates = Nx.broadcast(1, {Nx.flat_size(new_indices)})
        new_mask = Nx.indexed_put(new_mask, new_indices, new_updates)

        new_mask
      else
        mask
      end

    new_indices =
      mask |> Nx.flatten() |> Nx.cumulative_sum() |> Nx.subtract(1) |> Nx.to_list()

    mask_l = Nx.to_list(mask)

    {new_vocab, removed_terms} =
      Enum.reduce(vectorizer.vocabulary, {%{}, MapSet.new([])}, fn {term, old_index},
                                                                   {vocab_acc, removed_acc} ->
        case Enum.fetch!(mask_l, old_index) do
          1 ->
            {Map.put(vocab_acc, term, Enum.fetch!(new_indices, old_index)), removed_acc}

          _ ->
            {vocab_acc, MapSet.put(removed_acc, term)}
        end
      end)

    kept_indices = where_columns(mask)

    if Nx.flat_size(kept_indices) == 0 do
      raise "After pruning, no terms remain. Try a lower min_df or a higher max_df."
    end

    tf = Nx.take(tf, kept_indices, axis: 1)
    {tf, new_vocab, removed_terms}
  end

  @doc """
  Fits a CountVectorizer to a corpus.

  The CountVectorizer will build a vocabulary from the corpus using the `preprocessor` and `tokenizer` options.
  The vocabulary will then be modified according to the `max_features`, `min_df`, and `max_df` options, which
  effectively filter out features from the vocabulary. Those options must have been set when the CountVectorizer
  was created with `CountVectorizer.new/1`.

  If you wish to fit a CountVectorizer to a corpus and then transform a different corpus using the same vocabulary,
  you can use `CountVectorizer.fit/2` to fit the CountVectorizer to the first corpus and then use `CountVectorizer.transform/2`
  to transform the second corpus using the same vocabulary.

  If, however, you wish to fit a CountVectorizer to a corpus and then transform the same corpus using the same vocabulary,
  it is recommended to use `CountVectorizer.fit_transform/2` to fit the CountVectorizer to the corpus and then transform the corpus
  using the same vocabulary. You may still pipe the output of `CountVectorizer.fit/2` into `CountVectorizer.transform/2`, but
  `CountVectorizer.fit_transform/2` is more efficient since the Term-Framquency matrix is computed during the fitting process
  and is not recomputed during the transformation process.


  Returns the new CountVectorizer.
  """
  def fit(vectorizer = %__MODULE__{}, corpus) do
    vectorizer = build_vocab(vectorizer, corpus)

    n_doc = length(corpus)
    tf = _transform(vectorizer, corpus, n_doc)

    df = Scholar.Preprocessing.binarize(tf) |> Nx.sum(axes: [0])

    max_doc_count =
      if is_float(vectorizer.max_df), do: vectorizer.max_df * n_doc, else: vectorizer.max_df

    min_doc_count =
      if is_float(vectorizer.min_df), do: vectorizer.min_df * n_doc, else: vectorizer.min_df

    {_, new_vocab, removed_terms} =
      limit_features(vectorizer, tf, df, max_doc_count, min_doc_count, vectorizer.max_features)

    struct(vectorizer, vocabulary: new_vocab, pruned: removed_terms)
  end

  defp _transform(vectorizer = %__MODULE__{}, corpus, n_doc) do
    if is_nil(vectorizer.vocabulary) do
      raise "CountVectorizer must be fit to a corpus before transforming the corpus. Use CountVectorizer.fit/2 or CountVectorizer.fit_transform/2 to fit the CountVectorizer to a corpus."
    end

    num_chunks = max(1, div(length(corpus), System.schedulers_online()))

    corpus
    |> Enum.with_index()
    |> Enum.chunk_every(num_chunks)
    |> Flow.from_enumerables(max_demand: 10, min_demand: 1)
    |> Flow.map(fn
      {doc, doc_idx} ->
        filtered_tokens =
          vectorizer
          |> do_process(doc)
          |> Enum.frequencies()
          |> Enum.filter(&Map.has_key?(vectorizer.vocabulary, elem(&1, 0)))
          |> Enum.map(fn {token, count} ->
            index = Map.get(vectorizer.vocabulary, token)
            offset = index * 64
            {offset, count}
          end)
          |> Enum.sort_by(fn {off, _} -> off end)

        case filtered_tokens do
          [{first_offset, update} | tail] ->
            {t, _} =
              tail
              |> Enum.map_reduce(first_offset, fn {next_offset, upds}, previous_offset ->
                {{
                   previous_offset,
                   next_offset,
                   upds
                 }, next_offset}
              end)

            encoding =
              [{0, first_offset, update} | t]
              |> Enum.map(fn {previous, current, update} ->
                cond do
                  previous == current ->
                    <<update::64-native>>

                  previous == 0 ->
                    <<0::size(current - if(first_offset == 0, do: 64, else: 0))-native,
                      update::64-native>>

                  true ->
                    previous_size =
                      current - previous - 64

                    <<0::size(previous_size)-native, update::64-native>>
                end
              end)
              |> then(fn b ->
                current_num_elements = div(IO.iodata_length(b), 8)
                num_zeros = (map_size(vectorizer.vocabulary) - current_num_elements) * 64
                IO.iodata_to_binary([b, <<0::size(num_zeros)-native>>])
              end)

            {doc_idx, encoding}

          _ ->
            # If all items have been filtered out, return a 0 vector
            {doc_idx, <<0::size(map_size(vectorizer.vocabulary) * 64)-native>>}
        end

      _ ->
        nil
    end)
    |> Enum.to_list()
    |> List.keysort(0)
    |> Enum.map(&elem(&1, 1))
    |> IO.iodata_to_binary()
    |> Nx.from_binary(:s64)
    |> Nx.reshape({n_doc, :auto})
  end

  @doc """
  Given a `CountVectorizer`, transforms a corpus into a term frequency matrix. The `CountVectorizer` must have been
  fit to a corpus using `CountVectorizer.fit/2` or `CountVectorizer.fit_transform/2` before calling this function.

  Returns the term frequency matrix.
  """
  def transform(vectorizer = %__MODULE__{}, corpus) do
    n_doc = length(corpus)
    tf = _transform(vectorizer, corpus, n_doc)

    if vectorizer.binary do
      Nx.select(Nx.greater(tf, 0), 1, 0)
    else
      tf
    end
  end

  @doc """
  Given a `CountVectorizer`, fits the vectorizer to a corpus and then transforms the corpus into
  a term frequency matrix. This is equivalent to calling `CountVectorizer.fit/2` and then calling
  `CountVectorizer.transform/2` on the same corpus, but is more efficient since the term frequency
  matrix is computed during the fitting process and is not recomputed during the transformation process.

  Returns a tuple containing the new CountVectorizer and the term frequency matrix.


  """
  def fit_transform(vectorizer = %__MODULE__{}, corpus) do
    vectorizer = build_vocab(vectorizer, corpus)
    n_doc = length(corpus)
    tf = _transform(vectorizer, corpus, n_doc)

    df = Scholar.Preprocessing.binarize(tf) |> Nx.sum(axes: [0])

    max_doc_count =
      if is_float(vectorizer.max_df), do: vectorizer.max_df * n_doc, else: vectorizer.max_df

    min_doc_count =
      if is_float(vectorizer.min_df), do: vectorizer.min_df * n_doc, else: vectorizer.min_df

    {tf, new_vocab, removed_terms} =
      limit_features(vectorizer, tf, df, max_doc_count, min_doc_count, vectorizer.max_features)

    tf =
      if vectorizer.binary do
        Nx.select(Nx.greater(tf, 0), 1, 0)
      else
        tf
      end

    vectorizer = struct(vectorizer, vocabulary: new_vocab, pruned: removed_terms)
    {vectorizer, tf}
  end
end
