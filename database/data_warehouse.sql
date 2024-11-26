PGDMP  1                
    |            data_warehouse    16rc1    16rc1 ,               0    0    ENCODING    ENCODING        SET client_encoding = 'UTF8';
                      false                       0    0 
   STDSTRINGS 
   STDSTRINGS     (   SET standard_conforming_strings = 'on';
                      false                       0    0 
   SEARCHPATH 
   SEARCHPATH     8   SELECT pg_catalog.set_config('search_path', '', false);
                      false                       1262    26072    data_warehouse    DATABASE     �   CREATE DATABASE data_warehouse WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'English_Philippines.1252';
    DROP DATABASE data_warehouse;
                postgres    false            �            1259    26088    dim_customers    TABLE     �   CREATE TABLE public.dim_customers (
    customer_id integer NOT NULL,
    customer_name character varying(255),
    segment character varying(50)
);
 !   DROP TABLE public.dim_customers;
       public         heap    postgres    false            �            1259    26087    dim_customers_customer_id_seq    SEQUENCE     �   CREATE SEQUENCE public.dim_customers_customer_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 4   DROP SEQUENCE public.dim_customers_customer_id_seq;
       public          postgres    false    218                       0    0    dim_customers_customer_id_seq    SEQUENCE OWNED BY     _   ALTER SEQUENCE public.dim_customers_customer_id_seq OWNED BY public.dim_customers.customer_id;
          public          postgres    false    217            �            1259    26081    dim_products    TABLE     �   CREATE TABLE public.dim_products (
    product_id integer NOT NULL,
    product_name character varying(255),
    category character varying(100),
    sub_category character varying(100)
);
     DROP TABLE public.dim_products;
       public         heap    postgres    false            �            1259    26080    dim_products_product_id_seq    SEQUENCE     �   CREATE SEQUENCE public.dim_products_product_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 2   DROP SEQUENCE public.dim_products_product_id_seq;
       public          postgres    false    216                       0    0    dim_products_product_id_seq    SEQUENCE OWNED BY     [   ALTER SEQUENCE public.dim_products_product_id_seq OWNED BY public.dim_products.product_id;
          public          postgres    false    215            �            1259    26102    dim_regions    TABLE     �   CREATE TABLE public.dim_regions (
    region_id integer NOT NULL,
    country character varying(100),
    state character varying(100),
    city character varying(100),
    postal_code integer
);
    DROP TABLE public.dim_regions;
       public         heap    postgres    false            �            1259    26101    dim_regions_region_id_seq    SEQUENCE     �   CREATE SEQUENCE public.dim_regions_region_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 0   DROP SEQUENCE public.dim_regions_region_id_seq;
       public          postgres    false    222                       0    0    dim_regions_region_id_seq    SEQUENCE OWNED BY     W   ALTER SEQUENCE public.dim_regions_region_id_seq OWNED BY public.dim_regions.region_id;
          public          postgres    false    221            �            1259    26095    dim_time    TABLE     �   CREATE TABLE public.dim_time (
    time_id integer NOT NULL,
    order_date date,
    ship_date date,
    year integer,
    month character varying(50),
    day integer
);
    DROP TABLE public.dim_time;
       public         heap    postgres    false            �            1259    26094    dim_time_time_id_seq    SEQUENCE     �   CREATE SEQUENCE public.dim_time_time_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 +   DROP SEQUENCE public.dim_time_time_id_seq;
       public          postgres    false    220                       0    0    dim_time_time_id_seq    SEQUENCE OWNED BY     M   ALTER SEQUENCE public.dim_time_time_id_seq OWNED BY public.dim_time.time_id;
          public          postgres    false    219            �            1259    26109 
   fact_sales    TABLE     �   CREATE TABLE public.fact_sales (
    sales_id integer NOT NULL,
    product_id integer,
    customer_id integer,
    time_id integer,
    region_id integer,
    order_id character varying(50),
    quantity integer,
    sales double precision
);
    DROP TABLE public.fact_sales;
       public         heap    postgres    false            �            1259    26108    fact_sales_sales_id_seq    SEQUENCE     �   CREATE SEQUENCE public.fact_sales_sales_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 .   DROP SEQUENCE public.fact_sales_sales_id_seq;
       public          postgres    false    224                       0    0    fact_sales_sales_id_seq    SEQUENCE OWNED BY     S   ALTER SEQUENCE public.fact_sales_sales_id_seq OWNED BY public.fact_sales.sales_id;
          public          postgres    false    223            e           2604    26091    dim_customers customer_id    DEFAULT     �   ALTER TABLE ONLY public.dim_customers ALTER COLUMN customer_id SET DEFAULT nextval('public.dim_customers_customer_id_seq'::regclass);
 H   ALTER TABLE public.dim_customers ALTER COLUMN customer_id DROP DEFAULT;
       public          postgres    false    218    217    218            d           2604    26084    dim_products product_id    DEFAULT     �   ALTER TABLE ONLY public.dim_products ALTER COLUMN product_id SET DEFAULT nextval('public.dim_products_product_id_seq'::regclass);
 F   ALTER TABLE public.dim_products ALTER COLUMN product_id DROP DEFAULT;
       public          postgres    false    215    216    216            g           2604    26105    dim_regions region_id    DEFAULT     ~   ALTER TABLE ONLY public.dim_regions ALTER COLUMN region_id SET DEFAULT nextval('public.dim_regions_region_id_seq'::regclass);
 D   ALTER TABLE public.dim_regions ALTER COLUMN region_id DROP DEFAULT;
       public          postgres    false    222    221    222            f           2604    26098    dim_time time_id    DEFAULT     t   ALTER TABLE ONLY public.dim_time ALTER COLUMN time_id SET DEFAULT nextval('public.dim_time_time_id_seq'::regclass);
 ?   ALTER TABLE public.dim_time ALTER COLUMN time_id DROP DEFAULT;
       public          postgres    false    220    219    220            h           2604    26112    fact_sales sales_id    DEFAULT     z   ALTER TABLE ONLY public.fact_sales ALTER COLUMN sales_id SET DEFAULT nextval('public.fact_sales_sales_id_seq'::regclass);
 B   ALTER TABLE public.fact_sales ALTER COLUMN sales_id DROP DEFAULT;
       public          postgres    false    224    223    224                      0    26088    dim_customers 
   TABLE DATA           L   COPY public.dim_customers (customer_id, customer_name, segment) FROM stdin;
    public          postgres    false    218   N4       	          0    26081    dim_products 
   TABLE DATA           X   COPY public.dim_products (product_id, product_name, category, sub_category) FROM stdin;
    public          postgres    false    216   k4                 0    26102    dim_regions 
   TABLE DATA           S   COPY public.dim_regions (region_id, country, state, city, postal_code) FROM stdin;
    public          postgres    false    222   �z                 0    26095    dim_time 
   TABLE DATA           T   COPY public.dim_time (time_id, order_date, ship_date, year, month, day) FROM stdin;
    public          postgres    false    220   �z                 0    26109 
   fact_sales 
   TABLE DATA           v   COPY public.fact_sales (sales_id, product_id, customer_id, time_id, region_id, order_id, quantity, sales) FROM stdin;
    public          postgres    false    224   �z                  0    0    dim_customers_customer_id_seq    SEQUENCE SET     L   SELECT pg_catalog.setval('public.dim_customers_customer_id_seq', 1, false);
          public          postgres    false    217                       0    0    dim_products_product_id_seq    SEQUENCE SET     L   SELECT pg_catalog.setval('public.dim_products_product_id_seq', 6426, true);
          public          postgres    false    215                       0    0    dim_regions_region_id_seq    SEQUENCE SET     H   SELECT pg_catalog.setval('public.dim_regions_region_id_seq', 1, false);
          public          postgres    false    221                        0    0    dim_time_time_id_seq    SEQUENCE SET     C   SELECT pg_catalog.setval('public.dim_time_time_id_seq', 1, false);
          public          postgres    false    219            !           0    0    fact_sales_sales_id_seq    SEQUENCE SET     F   SELECT pg_catalog.setval('public.fact_sales_sales_id_seq', 1, false);
          public          postgres    false    223            n           2606    26093     dim_customers dim_customers_pkey 
   CONSTRAINT     g   ALTER TABLE ONLY public.dim_customers
    ADD CONSTRAINT dim_customers_pkey PRIMARY KEY (customer_id);
 J   ALTER TABLE ONLY public.dim_customers DROP CONSTRAINT dim_customers_pkey;
       public            postgres    false    218            j           2606    26086    dim_products dim_products_pkey 
   CONSTRAINT     d   ALTER TABLE ONLY public.dim_products
    ADD CONSTRAINT dim_products_pkey PRIMARY KEY (product_id);
 H   ALTER TABLE ONLY public.dim_products DROP CONSTRAINT dim_products_pkey;
       public            postgres    false    216            r           2606    26107    dim_regions dim_regions_pkey 
   CONSTRAINT     a   ALTER TABLE ONLY public.dim_regions
    ADD CONSTRAINT dim_regions_pkey PRIMARY KEY (region_id);
 F   ALTER TABLE ONLY public.dim_regions DROP CONSTRAINT dim_regions_pkey;
       public            postgres    false    222            p           2606    26100    dim_time dim_time_pkey 
   CONSTRAINT     Y   ALTER TABLE ONLY public.dim_time
    ADD CONSTRAINT dim_time_pkey PRIMARY KEY (time_id);
 @   ALTER TABLE ONLY public.dim_time DROP CONSTRAINT dim_time_pkey;
       public            postgres    false    220            t           2606    26114    fact_sales fact_sales_pkey 
   CONSTRAINT     ^   ALTER TABLE ONLY public.fact_sales
    ADD CONSTRAINT fact_sales_pkey PRIMARY KEY (sales_id);
 D   ALTER TABLE ONLY public.fact_sales DROP CONSTRAINT fact_sales_pkey;
       public            postgres    false    224            l           2606    26136     dim_products unique_product_name 
   CONSTRAINT     c   ALTER TABLE ONLY public.dim_products
    ADD CONSTRAINT unique_product_name UNIQUE (product_name);
 J   ALTER TABLE ONLY public.dim_products DROP CONSTRAINT unique_product_name;
       public            postgres    false    216            u           2606    26120 &   fact_sales fact_sales_customer_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.fact_sales
    ADD CONSTRAINT fact_sales_customer_id_fkey FOREIGN KEY (customer_id) REFERENCES public.dim_customers(customer_id);
 P   ALTER TABLE ONLY public.fact_sales DROP CONSTRAINT fact_sales_customer_id_fkey;
       public          postgres    false    4718    218    224            v           2606    26115 %   fact_sales fact_sales_product_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.fact_sales
    ADD CONSTRAINT fact_sales_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.dim_products(product_id);
 O   ALTER TABLE ONLY public.fact_sales DROP CONSTRAINT fact_sales_product_id_fkey;
       public          postgres    false    216    224    4714            w           2606    26130 $   fact_sales fact_sales_region_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.fact_sales
    ADD CONSTRAINT fact_sales_region_id_fkey FOREIGN KEY (region_id) REFERENCES public.dim_regions(region_id);
 N   ALTER TABLE ONLY public.fact_sales DROP CONSTRAINT fact_sales_region_id_fkey;
       public          postgres    false    224    4722    222            x           2606    26125 "   fact_sales fact_sales_time_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.fact_sales
    ADD CONSTRAINT fact_sales_time_id_fkey FOREIGN KEY (time_id) REFERENCES public.dim_time(time_id);
 L   ALTER TABLE ONLY public.fact_sales DROP CONSTRAINT fact_sales_time_id_fkey;
       public          postgres    false    224    4720    220                  x������ � �      	      x��}�v�Ȓ���Q�2��	��;�2���^��@$$�	$� 2$�>�����^ħԗ��� ���T�&B�Hp7���5�诫g6.�iY�+6(�<���b��E�2M�ԸX��l�.SC�T��i��i�F}ǲ�(O�U6�/�?�~��e�'v�L��Ej�=>fӔ���e����������̵|�;?��I:}^y�1F����W�Ŋ��Ɠ�O춘�9�+`W�ӳ�������p�Q��f9?'�2]��	�����{���/���3�&�ϓ��\k���"O��ݯ�t�����IW+��I���L���׎�eZ���bQ�7L<���C��te��v�N7S��(����ol�&9%��x�dy���/��y��e���|;6*�U򐧰R��,������xU���_b�"O�2��M�����d7��l �Y�f*i���䷮}"�2b�=��0K�)��/֋���9|g1}i��T�x<��Z��	^�
�L^'-�*p��)�]*VpKE˂���d1����"�7��X��l=g_��U
ϴ����ĸd�g�����%p�Q��%���5�k�\&Gp��U���0_�˟����L�U��7(�i	_��	�a�X*��/�o\�lXL׸�pp����9;0z�.6\�$+�<��ilou�>�&yHs��ڡq�������N2z�dk��exsd�/��n�y&�)�<����!����<�ue�Y�D�y,J��2`~��)�[���Z�&%��d��R���~	J����:V*�'�ӂ]/f�
V��kV�j<��wx��`9�1�e-�E6�`o�j��?��\�����������aǘPg&�ulo��#�|��뷔k�>J�8�U|'���q�oiY�1;��jm�3�fyb�;^n�n���k��?S��$E+�o��x�����ӕt�9�����p@2g�
-�-,<��<�����	�@����2�f��	�{���_Oi����am��@�/�����!�%|�	�F�s�ͤXk7;��&�ڇ��V����*Y<��$�r���m�{��+��}Q̵�݉�RG�J�������ߪ5"��D�F�]�tQ�߀��Ѿ,���Q\[(�����5�/��]���;H8�@�J����e�O���臚}r=8~�5���pN-v�~h�ƒ����9\�%��_-��T\?y�H�������L`1
t|�5��$�Ͼ$�.��F?��U�A��<_`��	8^��e�3����o��_Q ��e	���A��N��3�|�o��R��لA�����ht"�>s����b�^LA��0�gS|C���-�DZ�e�1��fp*�e���یΠY�6�Bҿ�L�w<Ax�)X]����Ii�.�����<��*�*� Վ����t�~e�9��l�2I/�L�,Ђ6t�	�yV��d��w�BN����nUS�"��d���*��<�[�Q���e��v��M�f/�:v���7��<)Q��/��:�"۷���V��`S��A� ���B����tK_�n��Q�xp�N�Gpko��=��[����˾Ph6h�l
�:^ؽ�J}^��>ӝp/R���h/Vo�t�ŷ�}p��ok�lCH�����%]��>u~�1x���&<�J�,�k���gpPѳ#5pQ��9�g|J_AN���+i�<7���ض|E�2^a`q�m��zN����kǯ�F�?\��=�a	���x�Wf��y��d#8q���Ca�.ɋ!��1M�mr��)N�U�Gk����}��������D��#��W<���W�M��*�B�A�l�y�gK81pӍâs�	l9ã1*r8wh�W`��g3>�Wn��:��#���=��v-�� G���p�/�lY�B����yP�7�}L7E���]o�F{����X�^�(t���#EK�G��d֠�)��c cΟ���,�����_H��������54�H�Qϵ��;���eJ���{J	�*zx��I�bul��j{��2>߳9�-9�x#�< ��i�܈��E>�u�
<�b<-S�00D�����-69��IZ�ȁ��,�'կY�w�ރcP����	�-a�{()x�P�u~��5G�)[�}g����-V�8�>�]�i�_ۢ'�����1]T V+�0��͗k<�����3���uD����G�H��s���R���"Q(�_k��ah|:�\���Yo`�`��kxNv�=%D�c0�+�L���mc�<za$��<��"�f�6I�V�i�YX�4Z7�5
Vl[�e9q�[\C�U�RNu���%�?4�{�,�<��Y��`��[�̜���������5��	���A� �?��Vǋ���6�nL�}} 7f�y��t��Ɯ�C��b]3�V�W�e�zW�eh����8r��VBR�͞I���;>+�~\<��@����W�_����ݜk�U�t�"g�N�*��t�_���;����W�+��Ce�7��G���3jV�˱3y���V�[��Pt��c,#[δ��bף`C�a����p��c��}��5xn�F�xx�� ܧT�NZ�U�Ǟ!�)*��b�.70�E�ë�^Ƥ�t�}�N�ā�g.Q �tQ�'�	���n������[��L�HI���#���q*���v���!U��Fu�v�w�Q���I8�y��Hw�l��9�����7��sr3����: �3��S�/�(j��=����.�ٜG�>&��Ś��Z0�ry�x˼��<�<���z&�]�t|���l�� K���P��lG����xv�C�d��a��t���8��LϷ ��z$��/�HG�-il�`�i�����l�9�vضe�g���}���<C��z�Pt�¶m+o��D*؎q?�2����ё �SΈ{���� 3����IW�8���X>�{l>�1��W<z�|�t~�m{/�(����=Lt���T�8M�'��>ٹ�>׎栘Q5�	N���s���Zݶ�ΰ�E�����`ui�3���;9�6I��:�-���3V�ŔWD%i�j)/ж�Z�h7"���֐|��B ۱����Y�Q�t��s�zT	�i��I����m;�2��Z�Д��֔Z�ڤ�ֵ�p��߈�P���z���4y�^s�gp�.O�OU�U�Q)ŪlǗ�;/�hV���+sy�S�N`���|���J�s���CP:�L�`�S��OF��xe������r���?Ed���}b�lZ��OI�<�<}������vbYl�����n�ZKM���/��eP�mZ&�+Ma������8-�Q�`eZ|U�&��-��������}Q~�/���� ��d�kŕ;�����-��}ì��<��)?��� �P���\��s���W�� c��ӌ�N�����;i�݈
Ң�(���z���u����FjD���a���O��4�rˠ��%�bt�039�j/��~�v��G��9�.����ɡ$��A�hb�#�lZ'*)��,�e�I��[�<H��樮�������?��Ω�W7Y����J�b����5�w%���sJ��	О������Q�?�o|��M�>��ע|��"t퀝�O����r��؁8 ;��M�L�=�"�`��xݏ�E�Z�vIB�6�;�;(��Z���Z�m����
�$e{,/6�\��@���tj6U���i�SK/G_�����d�o�O��B� �82���T���ܰcP]K�`
�6��̧��V"|D�\�)�gγ������j�ߓ�6{`��x�}�-@(yՓ�98I�9�b%��ul�pg0��C��iJ��N��Fo0����3����1���2�45Da��M�5ry�ӢX�C��KF���_SR�ϔt�K��+oSL��L�s2K��a�E�1\=�A"!،<�O6�����GL�|�Z�ݠ	�⸸d_��"�����B���O�$i�v)pAd�lƾ�%z#w%^�	ӻ��#�\��T�D-P.�����~�����O    xp���Z��\&y�����u	aR���YyxB����� �̤�wX'+*��)w]���*aFf��%F8��3ԛ�k���*�#� �n_.�@�&s*���N�Y
�:ֺ��6L�Z�%�+R�*pLQ���儹�+Z��ތ!bm��0�v��:1c�>��41N���Hn`�����ڙt��O'
��y�B�v80�X�*���H��uy�IX<H�V�J���@-�HC_�~_A�&�.�(_����Zq_�χ��քy·�7�e��J�F��ʢ,���v`��j{� �Ҭ]��qsf�$wȆ��ǵ&B�h��,
D��#��t<-���X����@�)��d��&K8h���_���(HܵQ���v�,#��8��R^O�;JJgBQ���{989��� 䶏��ّk Ӽ�� ד&�I�=�>h�0���ه��-�4� �Ү�*�<mۤ�7��oVt$vS��}�v�UӾ^��]�F$C��9��x�w��0��������&WI�0�K���@6ty��w1� �L�1	�<�v��S��h|)��k�
Q������ ;����A>�~}cW|O�]ϗ�J㩞���2��A�,a��RQ%�-9��FQ �������m�Ք��"����l|�0�9Ղ>v��8���ʔ�^]�=�����)ǌ�U��.�������w��T	��� l��Kx�g���&Z�1��$�ɥSP�n�LQpWͱ��������j !�-��pٿk���A���a��LE���f�!1�����c�ͽnQ�	�!_�"��19�/r����:��J&��U^,���+����%�Ф�^S����Y�P��_=RZ�֡�+�a�-�.r�o���@ժ�S1�� �FO��pnE�x���S$&.xva��.��U,;�06�� �za��U�0�A1�D1AkX�a�c[�"]Mz7���ׯu1a�pm+=,V��@(���b��Ҥ+d�oYV+�A
��rM�	�?km'�D�D�69�]ׂ��<��o{����ΘW$W:��#ZO�A���t��l���m
[�%X�$��J�_/�w�������~�N�쮺�c��Q-3r,�e�����࿣oц丂��5����ǀ�;�禜��h!pTa:nu�Ms����LtpKv��,��y�z(=��8<��@\�k��V0we��	��w���x!�QZ��p{��$�邕},&q�ATNE9���,�4��(e]ﭗ��ж�8�}���D9�0m����Q����vy�q+T�q�����/��6Be��+8N(�	<�GqpLy�q"��ћj�vã��M����Of���K۲/�?TGA�qm0���ٛ=C�;ٛ�0���l��+6 �P]�w�:����� [���^���"���VK�e-x��|�c��$5E�y_p%9T�qN�[�G�Bg�����Mk��Yg������	�_ǣ��G��Z�l�~�����"����ʬ�Y���)7T�SKz����%Q������xǍ��\1`��������h1����s��j�Wk9��ƹ�*'8^ݚ�;� ��m����;xS�����fة�`�b��=����y���L�a�W�*��`�X&��zg9�[")��%�ٔ璨@ Vf�� =F��b�=���.�b�-&s�g!.{�>��Be��VYDH�M�9���X6�D���+��{��&���n�YGs|��s����`�x�	NC���T�M���1�3����!�#�Q��K�tT	p�Վ�S*ҭ��W�����GԜB?]��k���>J�� ��:��������$�T��	� X;��#5,��x�XǏ��S
�e;!X�� *�}�����7���7B@��cr�@�-�����Wc���5�L��8�=u��'�@�	't�[�8����浃}1�- ~�=�ܟ��Ny��9�n��!���S�����͙��W��� ��)��`IO���œm��ƃ^'���',ƥ�~С�'���=�@�$�Q��Iw�to	g��[���Փt�t
��<;G2����6�(���٘n=t�������@Iq#xQ���V}7�B~��o�-��k��?h���kB������ݼ�x�� 6���}����=��������[aHٕk��N����c�eQs����E�mб`�kkx�:�KJE��t�ư�YG�����ݬxE'��:��u�+r����I��jM�'�Z���j�nlQ�Þ��mպc�c\�Oi:��b���xL�"p{ �2別Ke)RL�^�Ƽ'�jv>ߨ[Q��l�� )�1j���@�������&�G7P��~����tE>���6<v��ZY��]+z�&��{]f0
أ��"�MB;_���}�`'��,Qw��Kc
�%j�����([,��'�"�s{�L�j�p T����=!C�+P*}Ɠ}�8a�\A~|�Aƾ�&&���j �3���{����cL�<�S�I`.�9.�A��QК�!X���i
D\:�����@)��� � � `�ݮ�b��1)�$y���>����-�x��}���}��z��zR^�Pߘ��' ,��Z&�r!�i�aɒ'�}l�Â�>BR6X��a�"�+��O�YMdIo0��6���f�'�n�T��u�r:,��wj��1��W��v-W4�זj��l�3����3�s�]�k�o8�!Ե��?��M�.X�>�a�>�\תy��<O^��u��w���5���Z�Si�w!cvH�)��L,a�ER4P�Z:/�
ґ�N&U�(Fl\ۮu�m��2y�zp�+�����<�8�S�$p����}�c�aI�g�r=�9@�H�F��)�Ԗɵ]���FO����͍�k�!ۥ֯�;X��0�-:�W���[9�̒c[]�W�"myյe,t�{��+Mx�V��Ȩ,��4|E%�A�*���ﮭJ��S�DVM\aTT��W����Vh�Jf���\ǩ{zt=���-����ɜd)��5�Asz��nK�P�/�0��	�,�rpT�5�7�;�K�����E�g�Oks�JC�xhBDI�mN(a%p��f�։F2�D[�i_�v���Ր�Guj�N�p��#�ŵj���#6��f���~f`�XE��jɺL���N���r����-k��C�a��O�DPV�5�� ���P-F�/z�p�Z��s �?k]ۡI]�8�������JL����V��3XM7Y�^��xGּ�,���H���x�I\72n�sD��ހ����\%;m�  ��+V��E���a�˽�:x{oמ���>��+p�:4��H(�� Z�ֳ� �҉����-����s��i
���Gӧ��3Y���Ar=O�ɰX�|E�n�:�F �!>%+<��M;+�.q`� l�C�;o( #<����H=�!�r�=KQ��iH/��7p5�m�+׫��C]���"鸂���c�G��#t�:���
�(�����z��x�m�wpz
R6���ȼ��F�-����o+�wb���Q7u}��j�w��a�w�3�nl]�X�-��%�f�T�%[N���9TroZnr,WuD���6�aĪ���z*B�~�a=Dƙ��IK%�\@��U�/T����4�.�q]��i>��:>�Hv-�, [pr]%���A^M�ܡ�<��p��ҡ���P�!5��̳���~7eW8��(K)G��	9�d� EK	��x����E�Mr�_�a�۸I+d�{N�ri>k2)��KE��H��N��ߊr!k����6�5���['�K�n�n�͋�U�~xqsww�W�����D�x��1�&�#l`,ӂ5��� ��0���Y�5=��= Ζv�o+�������;�J�C�H�9�N�"����X�qQ��\⌱߷�']�� V-�:āZ"o$�?x�ܑ��JMz�VHN�(�|�Zݠj��v�z�+�����    e�#z#l9�u��8�����J�K솮у�]&,�0c�I��;��n��}@Zj��7�pA���w�7��3ZnOˎ�؅�!�[a�ݠ(�%1* ��O�Nǧ�6?/w{?�O������A/�td�FO؋*�Z��ܳ�֙	%���[�P�+(2��� ��N-W��]����?���	z%G�!��gs�-[�:7�����FΖ��l�r��)r:+l��t�Gޟg���3��9�NF��#�{݀�7p�'dn��!ǜ�0'��g��Bi}�PAp�l�Q�yp��^)sRfs���N'~Q��kž�	���l���������"5�|��Oq�`�w&�_w��2�?��s1Ox�NƝ����J��t��5�̶+���ߙ��o[��Q�;������nc��C������yh2y휸�6e�o�i��n�C�n�Q6q���u{��0$���m�
����zQ�"N�޸�ٖ<�]���V�<t1>�DZ9�H6'���E�"����l�Od/(����#�j[!��렽D<NQ����E	K��_���k���5?�ϋ<Pq�R"wL���.���c�.��>�ϝ>���������-�^L�i���Tp��=�'B�s@�7&O�]w5�N�g��:@X8e-J�.0z��S�U�"/��yB8�����p��O)7p֑�,8/ea�F�"��NZ��@�-�Qwqī;s��',2!c,�����?GR��Ng���7E��<�M�t�ch�br��,����)f	�s���� ��௮��6���E�9��M-����<�ݞ���s��Q]��-���hɡ��g{��y:��ϣ�%����-c��9.�8|0+�+i0K���H�@�";�QweV�i�����n1V�Xws���yD_��H��v��� �F�F�v�c��@` \X'���q(M����h<�x�E�����ۻ�=�5���&��qo�4�f3LIpŎ�Rtz�';�h"ԯ�л7�!+�����6�89Beپ����|t�q�i�,yd0�A��!G���z.N��y+ǈ�����֧�ߦ��G%�~"��Xc54��gӝ8'2&�矆�9?��J���'ڤ�/�7Jn��ߦ ���;��\�������d-:͵t�!�+y���;�N�/����<��}�mP���M�T+��T+�uD��'�s�4R�6�ufS�Êg͎"]u�s�:�]�2�v�l�N ��Wb�K�����������Պo>�*Q���w�Skg�8��h�'����m��AI�\Qz�d!���l	��z�2cp9�'<E�L֊H����7�D޿�ܳ^qzg�y6�߮%��\��U�Ն���q�z�N0s��	��|׷Wy^CҴ�B<I��vZ� �<�Q��8Lj׭��5�s�9��]?�{Zm����I��x�ա�y�jm�ٳ��'��vO���[m����F�6��z��7{s.˩����=�F@������Ĳ�P]��}"_-��ļi}����Sw	��}xjb����4�u��� ���qs�.ݶ]��i�y��`4��{����%V��F�y��*3^�I�z�e�z�K6���9>8P���!H��f;��,|J��"ֺO`��q�(/�*�ί�C0N/pY{�n�9��N�#�����l�4�<�p�������,-�W�XL�����ݗ�*���T�)N���
�<�����h� <4&�������xATOI¡&��8]�����Z5Z�n|�M���75�wڂ��w�c�ɽ�Q�
��#i#�O\k���qB�5��N�N';�x�xx$>���{�0O����0{�|F�e-�h���d��o������U=�a兡���7�A�����Jx 9�O�:�[��j�c�¸�Σ(Tx�Y#�r01YƸX��_�� �e9}��!w��H�	��L+�~ePOVѷ�y�%6�u�&�Ț��M#�9���Ś�_T�(�.�{Q�� B#D���7��F]�G�#W��'��6kܝve�fG����l/
꜐6��n[�r�`�2�Ⱥ+ga׾S�`�D�6�wԛ�i��H�_�#K<U.��O|�a���{xL���,�'��7,F�Q�z��ŋ�B�V��P�|��nlK�;�>�(����m4�Q�JQ�(CTj��^�M!ȿ�1E��0�}	.Ep�Г�2���hi5=�"�)Xjm���=�����"��.y1ľH�dCx��xp�?��RD6xW�����dT�D�
���n
���Km���}A��z�a'YG�6��\!���l�j'C6��fR�7����t��;�[����Gd�p�[2�Hc��eH��@��z~R��o�_�6֦�Z�-Wǎx�Y�R��Z�|?�wɅ�LB�ӆ�p��	�׳snƎ-!zYj"e���4�Ű���@VT;:�
��\V"%R�.M�J�����[���|���J0;�ke�d���Y��877�c�������~��d7=�~��s\�v�N�߶�yi|�tC@���K�v�������a�h��s�sx��m�svE�y�c*��K��tG�ޥ�R��*m3c��qշU��,o5���FL�i�(v΀[L.�1�IdB$a�Y�J��\�����v�}��Ӟ2�ڳ�͚�n؇��r_p'ASqg*�1��m��������1�/{��o�D�0��5��"�
q�aP�E������;5��>�L�����w�}.�x���Sl��z%J�1{�%gHF~�������,���"�0s����1/C�B�����;5Yި̪y"h���k<�pC<��N.����� _��(�p�����fFd �ݥ'�kX�W����jY�L����oc�ԧ��T�N�g���M�]#z:�*{Ǹ�~[)�P���f��m�ĕ^ �6��s���I����P>4����pI��B��!��=[t���.�E�^T���6j�z[���[sr{��o3�nlLH�����gHP�̰��� 5+�#�f�1��M�q�wl#w���9��}�ds�0C��u�Emޓ+9��������D����"
�E1��� ��m�w7�l�?d��1Z�����a��|Ob�>����m�1��u.�r�6z騸ܝOl����ȱ�Ȧ�NOO�J���vl��s��
Պ�wt2��<�,yr��޹�룅��s̳�'����x���t]�����|�68O'��!�\��6�Q}��0΂�?ݲ�H�m�;;Z|�X�=�D���вk�(����m�� �̰��^V�}�؟��s���d"�C3u}hޑ}h����4�z��#C���(oկ;o���A
��b��no���mP�e��R��-$;�,�b��2����\c�`�1S�6Ӡ��KE��X�7ޚ�~;�\�jИ����> H��H� �1�d<���^V�p�f��5jـ��2}��|@щ����h�������E�b��j�~v<���<#�r�'��5h���|/�5�ࡰ=��	�ZW��l����*����*@Z{�T��>���������'�}���/�.�F`ۢ��@T~8dh"j���~	z���*N�� 4<���[
�;Q������6�q�p� l��[����'i��#ۋ>�2���ꆏ�T98�t%���Zr�~�ɗf�8YG���Q?r��x��ʄ��0�8X��F�Aы;�ۏ<� �xNm+�5헣���55;�#L?"�I��]P��S_�F������H��sh�����*iYC�(��T�C���rŖ�:���f����N]v6�9iӚM���b�q*]x�O؄@C���tѷ%�q�'yŒ�TZA���m��d���NV�(I���'7G��WƵ�Q�!<���N���Jx�a�V�2��v����Bԑ�O�n� � ��N��W��B� &��,:��l[���]	�o=Dޭ��	�,䴛e��p�	갑h_��6q
,�bh&y!�lD�-m�w�Ma]7;�H�&uk�� ��\��!+0�    D���Uo�[�yG�b����]u)���u���s���)O�����K���[D��mS��$��[�X�x�8́���v6�]7}��Y`;��W��GNn
p�}���m�D���eEpW|6��f3�l��V?n0�0�
:������!O���=��B���$v|k�n���ơ���lL$a�s�������ǒ%o�<Hء������-0��;m>�s�A�5�7j�<
q�qM`��Т��Q ��!�Pg=t�������h�+�N0/4f�OJUs����6[4��x�F�q^ol$�pRx�"px�Q�tRC�GOp>�������5�v�\��|J��a �	b��[�pue~	D�M�!�k�����q�k��C��7��M�a���)[x��nm��q\�B8|�9ՙ��r�����|i=����_�N�hj��2	U��hc���d�%�[.�Z]����e�6��9  ��Ɖ=K���8�+m���ӄKy.%�}Xʸ��	��O�6����mbm��rk��p�A�ʣ�6�[{�������R�eK����ש�ܠ6@��F�+8�D�9���u.��Y����K��7��	��b�c�Y�HQi��J��=��2�����7]xV�]A�攲?"y+C�w�\O��N�b� ���P�r���[΁�ly�l�^;zZ���&A�D��y�vK�����6�^�M"�r��aWe�Ãs�ı�`��:ul� �\�:Y�ǐ	�,�q�A}��K�Ɇ�$g�[M��J���[�Sa72�I5m�� Ze1�h�����9��G6Q�tIw��5��
+˨�0(��� �����սk_
���|�L�`��1�4 }ۼ?Ws�9�`�|5���8V�er~S�H:ߝ��������cbے��)�Ћ�HD����l$\p����4w�	=�����U�T�C�qL8��4D̫�� ��fF0�$(x.�h	%*�} $OI�(� p$F�qr��4G��������{�b�\�yJ�O*e����9�`�_�r0َʩs��k@�hs:2�A4��W�[��(�)hE�y=� �a)�8��$����3���H@4�D�͸)ґ%2lo"�m��O��}ɪ5�<G�wDNiщg(괘B}��h��v���tz־ݡ��L���Cr�+`�T�:�o1]CNO���G��5�~;�_z��'���"8씓�v{`�{�e2c�H��<�ɡ��i곍u��A�:\�̪�,mHJc��$�9
�
CN�f��^|��T��N�� �L˒�N�,E\�`]~��m��Dk���][.� 2'�q>�`Ǌ�ڮn�Q��"d"�Q�O��R�9����=�{��Ht�_1����8������oN�Qd�k���O�-i��d�Fԙ9m����&������zd��Uƴ])
E�巌���+6�`ޟ�0t �%�F�W*����`�LA�GV1�kQ�j���~�1<#���	oH�q�o�kL��c'1����\�F���T;�uxc*��1;M��u��O�$oqo�v�1�-��m�ppS�2��Z�� v$��"R��P� cL���QU̶g��5g�5	�1�*��HP�";F���Ѣ�b_��7�8
,!�R�G-��q���1�q�`��c��Pu��PǍ	-���XÅ�`_WQ<�t��	�a/fS��w��i�P���Hg���ӝ�K���BK�F������[����e ]q��(�x��RMh%$�\����-��]�-�S�<P	K�f-�)���a��2n��\���K�<X���[�?��e:3�gZDz�"ٟɟ2��[g��X{C+jB�ڦꑩ��xX��}ߝ�-�f�AכTǝ��d\��DN�Ģ�f���5�	�w�ڽ~���3޸�����A���BZ?Dx(��IQN�	·C?�.�ⴆ���#ә�D���w8(7��3d9᝗�*�ܯ�q�7혧u/dZ�}���ޙDy���r�=%� @2�_��������X�Km#D!Z�uS�uٿ�1��w�f��xp��6�Ў�����%8XJ��q:V���F��g}e"�<r�|��u���O��2R.U:��IN�zVp+u�ǧ���9�3<t��� �&��7�����l����MJw���dI����4�,[Ȥa���C���^	�7�%��I_��s�J�/t�r˄{!�NMM��H�Я� :�q� �c��������N�Q��
�<ݩ�O�������1�U��-i�,)��T���t.�.Ba��؞c�as��A�꺆�(^�ɩ��v5�z�҈�=tc΄S��f\�����Z�i2g��F6�$:���p���FT1�!�������Z�S���.t�8tX�ЍD�� z�9f4lب��x��f��&�"�����3B��/Z*]�x51b�%��F�Em�n��l˕Dw �1�������x��;�T���&5�xU�)���޽/:����ªI���U�W����W2y n���	!Z�G6wx7Z���Ğ�lUs�q$r�h���?t���@Z�u�����vֱ��ā�����Y��Wvs>d�<7���Ԡ>�ѐ���d���}��b���E��wk��ʹ��W<&�m��8�����T��޶~G����qͶ	H�_��K��&��$��`g�@��rY?P,�)Ih�8D�:�&����{�B?Tu��_%b��kr����u�G��r
�ǆ��7��,�	eC~�8����?ۘ���!�X���e��M+{	'xDiC��!�]�"��B$r��
g&���Js�*��f+���L�wI����rB�@�ԍb+�y<����v�hܩ���^7�� r)DY�m�+�@p��{��)wB�)���A���U��@�N])=�����Y�US��06��G�MA\�{9O�n�	�wM�t���.͡�b�Z<N���a((����"Ns>�����^CG�ꀑaX� ��>���3�H!�b�=�Z���3��H&e��N�����jE�:.P�?�W�v2��0إ��vh������P��\T���(�����-v�DM���`��o��-�p��}B# �L�Y5Ej'wfY���
��j�,|��j���o���aT;��.�Fzt��>x���m�y$�}�b��4��|Ϗ����
����/�~�A���)�!IK���m;J�x6D��!�:����cRQc:�l�(�ٵ�ʾSI�7$��!�),��aߍ��_�r���k���ƨc��,���ck�����*vaǭD����u�w����b2��|������1�9P������G�� �d�܅�O)���%��G��\���Q��'�����@X@�7s����p�8�4ĂcuC�y�Zu?n�@} �V�-_�vj����cg�������B�ƨ-�v}H�����Q�q��}���]Y�x����z��X-���?,�K&�'< m�b��rk�Mڢ��:*�ڦ#�Q`���eF�1�:=ω���Ƽ렫E+¢��3�'H�����VV��HF8�k6�~�^�����u	��9��TW?9R��=a�oT5�2�S� o]Rа�߀��*p�q����E}���q1g���$C�����9�,���u��FUI�AXqIrp"�QV|D�з�-W�mC80���Wrn(WO�/OZR>b�mG0���q3���ǘ*4���ǕGv�}]�n�fpS�����o�,�'��w�H�.o��mf)�=0���c�'8�[�x����P��K1�L����И�/�u��
d���T��y�"r`ّ�B�n֍�XF���SNi��[k��-R�ԓvZ�E�2��(���S+���P��qI�{�P,�]@���R��	�*�Hެ��H�"�HjRT�Y�!G4髻9r�:��>(5r�+#Z\!}ح��>ݥ Nb�g�%�%BE	�8�{�[`�J�pg�F%���1�L"'2~����v46���h�-�Kʹ(V�q���ёRI�r���ǅ .  �k\N�%�P��� m�>�W�OC��B4!�n�c�	A�B���z�%�\��e��O�/��2SR&+Y5��F�~k�ApO�rՖ7(X�X��)��]���_��OUEm�\oٓm-��.��u�Ց��vP�����Q$�VW��*��9#7�X�Wp��}�#���a A&a����M�5�*����<!�.���R�T!oڳT]�]bbj��s�����}�Q� S���l�{brE^���Z乼}q�}����4�1��(\?��zu�3:8��¬�����K�<�	#�ds6������Q<�,�~Уo"/����X*w��"/�K��v��Hp~;H�ym��ގ�P�[�/n�������\�1>���F�e;6�?�iI��v#Ru) ߑ�������J�<6ǘ�.y�ʢD>�@��X�&��9�� z9�=c�H�f��3y��T0*/pR�'��xS#.-�49�яDĠ;�>8����=���J��2/2v�=%�.��N�u{c��i�n���G����Ҫٓ�%&����ٜH��8!��Y<�!Y��{.�,C.ߵU��:�k�
���M_�d�-v��mu�F��,����r��N]3
\�a�����S��>R�u�ɣ���i��Dk���gG�P3^C��-�&��TN�����:ʪQ�7�l�l�kET��<	X �K-�֌�5��y_Tdd(�MK�X:>AM�۩*��h�(����3	F�ef��7yp�O�2C�.�3&���	d�(\�ƴH~ڃnl��9�����D�J���i'Q�m����7x}�]Ł���K� ~rlKdV�ë����no��Bj��P�E8�����M�E���m�I[g�j��{�cE!o��#�V]�� ���)#ďiJ��Q�-�T}�P4;D��%�q��<�������z1��W�����n�g�߄�;�mӜ�OD�zL���ݣ��y_���l>8dE�1��������\���Tۋ����]��&��}�(�s���uW�"X_��@��cQ��QIE�*nݪ�Q{m#��b6R.�V��c�~�y̕(���$��q_���kl[c��0;�#5h*�5� A?8w#�y!�����`ʶЋ�X�Y��G�+�#�.Z���T`���Q�s��
s=� ��-���9�I$��镴��bK��B�#�L�����6َ�`�fimgU[����q�C�<����{ � ���c���c�cKͮ��L�H5��a+�ӂ�6W��<r4P�jY�z���q$x?�x/���t����-D��E�Xw�M����&1�+�Vc��-~b1����t����������+Du�r�N�b�4Q���QU�M����V07#z���3O��#Ǎ����:(Bl)B'�
.�n�ٌmk�o�Ǌ�mMD��sZ����mGS���D2�X��h�+1��\֜�خ��N{���7����O��ȹFi�h�^KRG40�ʍ��e��#G|_#L�\`D(�r�Q���$Yn�FSNs0�	�;��N[ۑ�x��u�����/���q�            x������ � �            x������ � �            x������ � �     